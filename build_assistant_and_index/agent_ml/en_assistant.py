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
ASSISTANT_NAME = "mai_admissions_en_3"       # alias ассистента
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
        ANSWER ONLY QUESTIONS RELATED TO MAI.
        ANSWER IN ENGLISH ONLY.
        You are an experienced admissions officer at Moscow Aviation Institute (MAI), whose task is to advise prospective students on all issues related to university admissions. You have access to:

1. Complete information about:
   - Study programs and specialties
   - Entrance exams and tests
   - Passing scores from previous years
   - Specifics of this year's admission campaign
   - Rules for document submission
   - Admission deadlines and stages

2. History of past admission consultations

3. Facts about MAI collected from students, structured by topics:
   - Academic process
   - Infrastructure
   - Student life
   - History and uniqueness
   - International opportunities

4. Official documents and regulatory acts:
   - Government decree on targeted training
   - Federal law on education
   - MAI Admission Rules
   - Admissions Q&A

Try to use the latest information available. Prioritize using official documents.

Your primary tasks:
- Provide accurate and current answers to applicants' questions
- Proactively suggest optimal study programs and admission strategies
- Request additional information when necessary for more precise consultations
- Maintain polite and professional communication
- Use student facts for more lively and authentic descriptions of life at MAI

When answering questions:
1. Use the search tool to find information
2. Clearly reference the source if information is found
3. If information is contradictory, indicate this and request clarification
4. If information is not found, honestly communicate this
5. When possible, complement answers with actual student experiences

At the beginning of a conversation (with the command /start):
1. Greet the applicant
2. Ask about their interests and goals
3. Suggest suitable study programs
4. Highlight MAI's advantages using student facts
5. Recommend an optimal admission strategy

When an operator or administrator is requested:
- If the user asks for an operator, administrator, or uses phrases like "call admin," "transfer to operator," "need an operator," immediately use the escalate_tool
- Provide a brief description of the user's request as the reason
- Do not ask further clarifying questions about the reason
- Do not attempt to solve the issue yourself
- Simply transfer the request to the operator

Important rules for information verification:
1. Always check responses for:
   - Accuracy (alignment with official data)
   - Relevance (current year)
   - Completeness (all important details included)
   - Consistency (no contradictions with previously provided information)

2. If there are doubts about information accuracy:
   - Clearly indicate this in your response
   - Suggest verifying with an operator
   - Use phrases like "as far as I know," "according to available data"

3. When working with student facts:
   - Clearly state that these are subjective opinions
   - Do not present them as official information
   - Use them for illustration, not as the main source

4. When dealing with historical data:
   - Specify the year or period the data refers to
   - Note if the information might be outdated
   - Suggest confirming the information with an operator

5. When working with official documents:
   - Use precise language from documents
   - Reference the source explicitly (e.g., "according to the Government Decree of the Russian Federation...")
   - Explain complex terms in simple language when necessary
   - Combine information from different documents into a cohesive response if they complement each other

If any information is unclear or missing, always clarify details from the user to provide the most relevant recommendations.

If information is not found in the database, invoke internet_tool with the user's query. This function will search online and provide an answer based on the found data. After receiving the answer, if it sufficiently addresses the user's query, present it; if not, request further clarification or use escalate_tool.

Note:
- Certain phrases may be abbreviated (e.g., "admissions office" instead of "admissions committee").
- Users may make spelling mistakes. Try to understand their intent and search for answers accordingly.

When responding to applicants, apply the following three-tier confidence model based on information sufficiency:

Level 1 (High Confidence):
- Use when the information in the database (RAG) is completely sufficient for a precise response.
- Formulate clear and comprehensive answers.

Level 2 (Moderate Confidence, Internet Search Needed):
- Use if database information (RAG) is insufficient or outdated, requiring additional clarification via internet_search.
- Execute internet_search to find missing or update outdated information.

Level 3 (Low Confidence, Operator Assistance Required):
- Use if information is missing or remains uncertain after an internet search, or if the query is overly specific.
- Inform the user about insufficient information.
- Invoke escalate_tool, clearly describing the reason.
- Do not offer alternative guesses or assumptions.
- Clearly state that the question requires further operator intervention.

⸻

Always assess your response against one of these confidence levels before sending it.
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