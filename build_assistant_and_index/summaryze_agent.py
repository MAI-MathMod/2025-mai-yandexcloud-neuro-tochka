#!/usr/bin/env python3
"""
MAI Thread Summarizer (без RAG)
==============================

Одношаговый агент, который принимает поток сообщений (тред) и по запросу
выдаёт компактное резюме беседы: кто задавал вопросы, какие ответы получили,
о чём договорились, что делать дальше.

* **Без RAG** — никакого поиска по базе знаний, чистая генерация на основе
  истории сообщений.
* **LLM‑инструкция** задаёт формат итогового вывода (5–7 пунктов).
* Команда **/summary** в консоли запускает генерацию резюме.

Запуск:
$ python mai_thread_summarizer.py

Требуется:
    pip install --upgrade "yandex-cloud-ml-sdk>=0.3"
Переменные окружения:
    YANDEX_FOLDER_ID, YANDEX_API_KEY
"""

import os, datetime as dt
from yandex_cloud_ml_sdk import YCloudML

# ─── Константы ────────────────────────────────────────────────
ASSISTANT_NAME = "mai_thread_summarizer4"
FOLDER_ID = "b1gst3c7cskk2big5fqn"  # ID папки в Yandex Cloud
API_KEY   = "AQVNzzJielnSayrAOlQWlxDMK49OShvzdqtUQdAp"  # API-ключ

if not FOLDER_ID or not API_KEY:
    raise RuntimeError("Экспортируйте YANDEX_FOLDER_ID и YANDEX_API_KEY")

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)

# ─── 1. Создание / поиск ассистента‑суммаризатора ─────────────

def get_or_create_summarizer():
    existing = [a for a in sdk.assistants.list() if a.name == ASSISTANT_NAME]
    if existing:
        print(f"✔ Суммаризатор найден: {existing[-1].id}")
        return existing[-1]

    print("◆ Создаём суммаризатор…")
    system_prompt = (
        """
Вы — менеджер приёмной комиссии МАИ. Вам передают историю диалога между
абитуриентом (и/или его представителем) и приёмной комиссией.

Сформируйте расширенное резюме на русском языке, придерживаясь следующей
структуры (без Markdown):
1. Участники и корректное обращение к каждому (имя, Вы/ты, канал связи — если
   упоминается).
2. Ключевые академические и карьерные интересы кандидата.
3. Основные вопросы / опасения, поднятые абитуриентом.
4. Сводка предоставленных ответов, ссылок и документов.
5. Важные дедлайны и логистические детали (экзамены, подача документов,
   встречи, оплаты и т.д.).
6. Нерешённые или открытые пункты, требующие уточнения.
7. Эмоциональный тон и мотивация кандидата (одним предложением).
8. Проактивные идеи для следующего шага со стороны приёмной комиссии
   (2–4 конкретных действия), например:
   • отправить персональную подборку программ или олимпиад;
   • пригласить на ближайший День открытых дверей или экскурсию;
   • предложить консультацию по телефону / видеосвязи;
   • выслать чек‑лист документов и напоминание о дедлайнах;
   • подключить к профильному студенческому чату или ментору.

Не добавляйте фактов, которых нет в сообщениях, но смело предлагайте
релевантные проактивные действия, исходя из контекста диалога.
Объём — 8‑10 нумерованных пунктов, короткие ясные фразы.
"""
    )

    return sdk.assistants.create(
        model="yandexgpt",
        temperature=0.3,
        name=ASSISTANT_NAME,
        ttl_days=30,
        expiration_policy="since_last_active",
        instruction=system_prompt,
        tools=[],  # никаких RAG‑инструментов
    )

# ─── 2. Консольный диалог с командой /summary ────────────────

def chat_loop(assistant):
    print("Введите сообщения треда. Команда /summary — получить резюме, exit — выход.")
    thread = sdk.threads.create(ttl_days=1, expiration_policy="static")
    try:
        while True:
            line = input("💬 > ").strip()
            if not line:
                continue
            cmd = line.lower()
            if cmd in {"exit", "quit", "выход"}:
                break
            if cmd == "/summary":
                # Запрашиваем резюме у ассистента
                run = assistant.run(thread)
                res = run.wait()
                print("\n📄 Резюме:\n" + (res.text or "<пусто>").strip() + "\n")
                continue
            # Иначе считаем, что это новое сообщение в треде от пользователя/канд.
            thread.write(line)
    finally:
        thread.delete()

# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    summarizer = get_or_create_summarizer()
    chat_loop(summarizer)
