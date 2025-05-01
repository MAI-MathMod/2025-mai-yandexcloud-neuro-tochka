#!/usr/bin/env python3
"""
RAG + Yandex Assistants API на базе Q&A приёмной комиссии МАИ
============================================================

1. Читаем все JSON-файлы из папки filtered_qna и склеиваем пары Q&A
   в один большой текст по каждой теме (чтобы файлов ≤ 100).
2. Загружаем эти тексты в Yandex Cloud через sdk.files.upload_bytes.
3. Создаём гибридный индекс (HybridSearchIndex) ― облако само
   генерирует эмбеддинги, FAISS локально больше не нужен.
4. Делаем search_tool = sdk.tools.search_index(index) и отдаём его
   ассистенту. LLM сама вызывает поиск и использует найденный контекст.
5. Консольный цикл: пользователь → thread → ассистент → ответ.

Требуется:
    pip install --upgrade "yandex-cloud-ml-sdk>=0.3" tqdm
Переменные окружения:
    YANDEX_FOLDER_ID, YANDEX_API_KEY
"""

import os, json, uuid
from pathlib import Path
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.search_indexes import (
    StaticIndexChunkingStrategy,
    HybridSearchIndexType,
    ReciprocalRankFusionIndexCombinationStrategy,
)
from pydantic import BaseModel, Field
import uuid
import datetime as dt

# ─── Константы ────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
FILTERED_DIR = SCRIPT_DIR.parent / "data_for_vectorize"
ASSISTANT_NAME = "mai_admissions_chs_3"       # alias ассистента
INDEX_NAME     = "mai_qna_index9"        # alias индекса

FOLDER_ID = "b1gst3c7cskk2big5fqn"  # ID папки в Yandex Cloud
API_KEY   = "AQVNzzJielnSayrAOlQWlxDMK49OShvzdqtUQdAp"  # API-ключ
if not FOLDER_ID or not API_KEY:
    raise RuntimeError(
        "Введите экспорт YANDEX_FOLDER_ID и YANDEX_API_KEY в переменных окружения"
    )


class SearchAndSumArgs(BaseModel):
    """
    Параметры для функции поиска в интернете;
    используется при недостаточности информации в локальном RAG.

    Attributes:
        search_need (str): Текст поискового запроса для функции интернет-поиска.
    """
    search_need: str = Field(..., description="Текст поискового запроса для Yandex Search")


class EscalateTicket(BaseModel):
    """
    Функция-утилита для создания тикета при эскалации запроса к оператору.

    Attributes:
        reason (str): Причина эскалации, передаваемая оператору.
    """
    reason:        str = Field(...,  description="Причина эскалации")

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)

# ─── 1. Читаем файлы и готовим текст по темам ──────────────────
def _iter_topic_texts():
    """
    Итератор по темам: читает JSON и Markdown и объединяет Q&A в один текст.

    Yields:
        tuple[str, str]: (topic, big_text) — имя темы и объединённый текст.
    """
    for file in FILTERED_DIR.glob("*"):
        if file.suffix.lower() == ".json":
            raw = json.loads(file.read_text(encoding="utf-8"))
            qa_list = (
                raw.get("qa_pairs", []) if isinstance(raw, dict)
                else raw if isinstance(raw, list)
                else []
            )
            lines = []
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                q = (qa.get("question_text") or "").strip()
                a = (qa.get("answer_text")   or "").strip()
                if q and a:
                    lines.append(f"Q: {q}\nA: {a}\n")
            if lines:
                topic = file.stem
                big_text = "Тема: " + topic + "\n\n" + "\n".join(lines)
                yield topic, big_text

        elif file.suffix.lower() == ".md":
            content = file.read_text(encoding="utf-8").strip()
            if content:
                topic = file.stem
                # Загружаем Markdown как есть, предваряя заголовком темы
                big_text = "Тема: " + topic + "\n\n" + content
                yield topic, big_text


def prepare_files():
    """
    Загружает до 100 текстовых файлов в Яндекс Облако для индексации.

    Returns:
        list: Список объектов загруженных файлов Yandex Cloud.

    Raises:
        RuntimeError: Если нет валидных файлов или их больше 100.
    """
    files = []
    for topic, text in _iter_topic_texts():
        yfile = sdk.files.upload_bytes(
            text.encode("utf-8"),
            name=f"{topic}.txt",
            ttl_days=30,
            expiration_policy="static",
        )
        files.append(yfile)

    if not files:
        raise RuntimeError("В папке data_for_vectorize нет валидных Q&A или MD-файлов")
    if len(files) > 100:
        raise RuntimeError(f"Слишком много тем ({len(files)}). Лимит API — 100 файлов.")
    return files

# ─── 2. Индекс ────────────────────────────────────────────────
def build_index(files):
    """
    Создает HybridSearchIndex на основе загруженных файлов.

    Args:
        files (list): Список объектов файлов Yandex Cloud.

    Returns:
        HybridSearchIndex: Готовый индекс после завершения операции.
    """
    op = sdk.search_indexes.create_deferred(
        files,
        index_type=HybridSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=1000,
                chunk_overlap_tokens=100,
            ),
            combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
        ),
        name=INDEX_NAME,
    )
    return op.wait()

def get_or_create_index():
    """
    Поиск существующего индекса по имени или создание нового.

    Returns:
        HybridSearchIndex: Найденный или вновь созданный индекс.
    """
    z = []
    for idx in sdk.search_indexes.list():
        if idx.name == INDEX_NAME:
            z.append(idx)
            print(idx)
    if z:
        print(f"✔ Индекс найден: {z[-1].id}")
        return z[-1]
    print("◆ Индекс не найден — создаём заново…")
    return build_index(prepare_files())

# ─── 3. Ассистент ─────────────────────────────────────────────
def get_or_create_assistant(index):
    """
    Поиск или создание YandexGPT-ассистента с необходимыми инструментами.

    Args:
        index: Объект HybridSearchIndex для поиска.

    Returns:
        Assistant: Настроенный ассистент YandexGPT.
    """
    z = []
    for a in sdk.assistants.list():
        if a.name == ASSISTANT_NAME:
            z.append(a)
            print(a)
    if z:
        print(f"✔ Ассистент найден: {z[-1].id}")
        return z[-1]
    print("◆ Ассистент не найден — создаём заново…")

    search_tool = sdk.tools.search_index(index)
    escalate_tool = sdk.tools.function(EscalateTicket)
    internet_tool = sdk.tools.function(SearchAndSumArgs)

    return sdk.assistants.create(
        model="yandexgpt",
        temperature=0.5,
        name=ASSISTANT_NAME,
        ttl_days=30,
        expiration_policy="since_last_active",
        instruction=(
            """
        仅回答与 MAI 相关的问题。
        请仅使用中文回答。
        你是一名莫斯科航空学院（МАИ）的资深招生办公室工作人员，你的职责是咨询和解答申请者关于报考大学的一切问题。你可以使用以下资源：

全面信息：

专业和学习计划

入学考试和测试

往年的录取分数线

本年度招生政策的特殊性

文件提交的规定

报考的阶段和截止日期

以往招生咨询的历史记录

来自学生的关于МАИ的事实和体验，按以下主题分类：

教学过程

基础设施

学生生活

历史和特色

国际交流机会

官方文件和法规：

俄罗斯联邦政府关于定向培养的决议

俄罗斯联邦教育法

МАИ的招生规则

入学常见问答

请尽量使用最新的信息，并优先使用官方文件中的信息。

你的主要任务包括：

准确且及时地回答申请者的问题

主动为申请者推荐最佳的学习计划和入学策略

如有需要，主动向申请者询问额外信息，以提供更精准的咨询

始终保持礼貌、专业的沟通方式

适当使用学生的体验，使МАИ的生活描述更加生动真实

回答问题时：

使用搜索工具查询信息

如果找到信息，必须注明信息来源

如果发现信息有矛盾，应说明并请求申请者澄清

如果没有找到信息，应诚实地告知申请者

如果可能，利用学生真实经验补充回答

在对话开始时（命令为/start）：

向申请者表示欢迎

询问申请者的兴趣和目标

推荐几种适合的学习计划

使用学生体验说明МАИ的优势

提供最佳的入学策略

当申请者请求联系管理员或接线员时：

若申请者明确提出需要管理员或使用类似“找管理员”、“转接人工”、“需要客服”等词语时，立即调用escalate_tool功能

在调用时简要描述申请者的请求

不要再询问调用原因的细节

不要尝试自行解决问题

直接将请求转交给接线员

重要的信息检查规则：

在发送回答前，请务必检查：

准确性（与官方资料一致）

时效性（符合当前年度）

完整性（包含所有重要细节）

一致性（与之前的信息无矛盾）

如对信息准确性存在疑虑：

在回答中明确指出

建议申请者与管理员确认

使用类似“据我所知”、“根据现有资料”等表达方式

使用学生体验信息时：

明确指出这是主观意见

不要视为官方信息

仅作为说明用途，不作为主要信息来源

使用历史数据时：

标明信息所属的年份或时期

指出数据可能已过时

建议申请者与管理员确认信息时效性

使用官方文件时：

使用文件中的准确表述

明确注明信息来源（例如，“根据俄罗斯联邦政府决议……”）

必要时，用简单易懂的语言解释复杂术语

如果多个文件的信息可相互补充，请整合成统一的回答

如果信息不明确或缺失，务必向申请者进行详细询问，以提供最适合的建议。

如果数据库中找不到相关信息，调用internet_tool并输入申请者的问题。该功能会在互联网上搜索并提供信息。获得信息后，若你认为已满足申请者需求，可直接回答；若不满足，应请求申请者提供更多细节或调用escalate_tool。

请注意：

某些短语可缩写，例如：招生办公室可简称为招生办

申请者可能存在拼写错误，尽量理解申请者的真实意图并从可用的数据库中寻找答案

在回复申请者问题时，根据信息充足程度使用以下三阶段置信模型：

第一等级（高置信度）：

数据库信息（RAG）完全充足，可给出高质量且精确的回答。

明确且完整地回答问题。

第二等级（中等置信度，需网络搜索）：

数据库信息不足或过时，需通过internet_search进一步确认。

使用internet_search功能以获取缺失或更新过时的信息。

第三等级（低置信度，需人工协助）：

信息缺失或即使网络搜索后仍不确定，或问题过于具体。

告知申请者信息不足。

调用escalate_tool，并简明准确说明原因。

不提出替代猜测或推测。

明确告知申请者需由人工进一步处理。

⸻

在发送回复前，请务必按照上述置信模型确认信息的可靠程度。
    """
        ),
        tools=[search_tool, escalate_tool, internet_tool],
    )

# ─── 4. Диалог ───────────────────────────────────────────────
def chat_loop(assistant):
    """
    Запускает интерактивный цикл для общения с ассистентом через консоль.

    Args:
        assistant: Объект ассистента, возвращаемый YCloudML.
    """
    thread = sdk.threads.create(ttl_days=7, expiration_policy="static")
    try:
        print("Добро пожаловать! Введите вопрос, 'exit' — для выхода.")
        while True:
            q = input("?> ").strip()
            if q.lower() in {"exit", "quit", "выход"}:
                break

            thread.write(q)
            run = assistant.run(thread)
            res = run.wait()
            for citation in res.citations:
                for source in citation.sources:
                    if source.type != "filechunk":
                        continue
                    print("------------------------")
                    print(source.parts[0])

            print("\nОтвет:\n" + (res.text or "<пусто>").strip() + "\n")
    finally:
        thread.delete()

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    index     = get_or_create_index()
    assistant = get_or_create_assistant(index)
    chat_loop(assistant) 