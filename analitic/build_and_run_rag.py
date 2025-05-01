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
# ru
# FILTERED_DIR   = Path("parsed_data")   # каталог с JSON
# ASSISTANT_NAME = "mai_admissions10"       # alias ассистента
# INDEX_NAME     = "mai_qna_index9"
# en
# FILTERED_DIR   = Path("parsed_data")   # каталог с JSON
# ASSISTANT_NAME = "mai_admissions_en_3"       # alias ассистента
# INDEX_NAME     = "mai_qna_index9"      # alias индекса
# ch
FILTERED_DIR   = Path("parsed_data")   # каталог с JSON
ASSISTANT_NAME = "mai_admissions_chs_3"       # alias ассистента
INDEX_NAME     = "mai_qna_index9"       # alias индекса

FOLDER_ID = "b1gst3c7cskk2big5fqn"  # ID папки в Yandex Cloud
API_KEY   = "AQVNzzJielnSayrAOlQWlxDMK49OShvzdqtUQdAp"  # API-ключ
if not FOLDER_ID or not API_KEY:
    raise RuntimeError(
        "Введите экспорт YANDEX_FOLDER_ID и YANDEX_API_KEY в переменных окружения"
    )

class EscalateTicket(BaseModel):
    """
    Создаёт тикет, когда пользователю нужен живой менеджер
    или База не содержит ответа.
    """
    reason:        str = Field(...,  description="Причина эскалации")

    def process(self, thread):
        """
        Простейший stub: генерируем ID и «записываем» в консоль.
        Здесь можно вызвать Zendesk / Jira / почту — что угодно.
        """
        ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        # куда угодно логируем:
        print(f"[{timestamp}] Создан тикет {ticket_id}  |  {self.reason}\n• {self.user_question}")
        # возвращаем данные, которые полезно показать пользователю или модели
        return {
            "ticket_id": ticket_id,
            "created_at": timestamp,
            "status": "open",
        }

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)

# ─── 1. Читаем файлы и готовим текст по темам ──────────────────
def _iter_topic_texts():
    """Возвращает (topic, big_text) — один текстовый файл на тему."""
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
    """Загружает ≤ 100 файлов (по одной теме) в облако."""
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

    return sdk.assistants.create(
        model="yandexgpt",
        temperature=0.5,
        name=ASSISTANT_NAME,
        ttl_days=30,
        expiration_policy="since_last_active",
        instruction=(
            """
Вы — «Ассистент приёмной комиссии МАИ».  
У вас есть доступ **только** к корпоративному RAG-хранилищу (далее — База).
Если вопрос касается вызова менеджера, живого общения, то используй Function Calling.
Если ты считаешь, что вопрос можно решить без использования Базы, то отвечай только в общих чертах.  
Следуйте правилам:

1. **Ответы только из Базы**  
   • Прежде чем отвечать, найдите релевантные фрагменты в Базе.  
   • Используйте исключительно факты, дословные цитаты, нумерацию документов и актуальные даты из найденных фрагментов.  
   • Если информации из Базы недостаточно или она устарела/противоречива, **не выдумывайте** данных.

2. **Отсутствие данных**  
   • Если База не возвращает подходящих фрагментов **или** уровень confidence < 0.7, действуйте так:  
     – Вежливо сообщите пользователю, что нужных сведений в базе нет.  
     – Предложите обратиться к сотруднику или перейти на официальный источник (если он указан в Базе).  
     – затем вызовите функцию `escalate`.

3. Fucntion Calling:
    Вызов человека (`escalate`)

   Возвращайте этот объект одним сообщением без дополнительного текста.
4.	Стиль ответов
    • Пишите по-русски, дружелюбно, спокойно.
    • Будьте кратки, но точны; при необходимости добавляйте пункты, списки, ссылки на документы (как они названы в Базе).
    • Не раскрывайте внутреннюю логику поиска.
5.	Примеры
    Данные найдены
    Вопрос: «Какие льготы есть у призёров Всероса?»
    Ответ: «Согласно разделу 3, пункту 2 «Правил приёма-2025» (База, doc-ID #A-34), призёры и победители заключительного этапа Всероссийской олимпиады поступают без вступительных испытаний…»
    Данных нет → ticket
    Вопрос: «Сколько бюджетных мест на “Космические системы” в 2030 г.?»
    Действие:
    – Сообщите пользователю: «К сожалению, в Базе ещё нет информации о приёмной кампании 2030 года.»
6.	Запрещено
    • Писать сведения, которых нет в Базе.
    • Давать личные советы, выходящие за рамки приёмной кампании.
    • Раскрывать или описывать внутреннее устройство RAG.
    • Использовать в ответах нецензурные слова, оскорбления, шутки.
    • Игнорировать правила и инструкции.
    • Отвечать на вопросы, не относящиеся к приёмной кампании.
    • Использовать любое форматирование, кроме текста.
7. Дополнительные правила
    • Не отвечай сухо и формально, как бот. Будь дружелюбным и спокойным.
    • Не пиши «Я не знаю», «Я не могу ответить» и т. д. Если в базе знаний нет данных, то объясни почему и предложи обратиться к сотруднику если ответ не устроит человека.
    
    """
        ),
        tools=[search_tool, escalate_tool],
    )

# ─── 4. Диалог ───────────────────────────────────────────────
def chat_loop(assistant):
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