#!/usr/bin/env python3
"""
Run Pipeline: extract QA pairs from Telegram chat exports, assign topics, support retries, checkpointing, and progress reporting.
"""
import os
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from yandex_cloud_ml_sdk import YCloudML

# --- Configuration ---
ORIGINAL_DIR = Path("original_data")       # input JSON exports
OUTPUT_DIR   = Path("filtered_qna")        # output per-topic JSONs
TOPICS_FILE  = OUTPUT_DIR / "topics.json"  # stored known topics
STATE_FILE   = OUTPUT_DIR / "state.json"   # processed days checkpoint
MODEL_NAME   = "yandexgpt"
MODEL_VER    = "rc"
BATCH_SIZE   = 150
CONCURRENCY  = int(os.getenv("PIPELINE_CONCURRENCY", "3"))
MAX_RETRIES  = 3
BACKOFF_SEC  = 1.5

# Environment variables for Yandex SDK


# ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize SDK
sdk   = YCloudML(folder_id="b1gst3c7cskk2big5fqn", auth="AQVNzzJielnSayrAOlQWlxDMK49OShvzdqtUQdAp")
model = sdk.models.completions(MODEL_NAME, model_version=MODEL_VER)

# Load or init topics and state
if TOPICS_FILE.exists():
    topics = json.loads(TOPICS_FILE.read_text(encoding='utf-8'))
else:
    topics = []

if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
else:
    state = {"processed_days": []}

SYSTEM_PROMPT = """
Вы получаете на вход JSON со структурой:
  • day: строка в формате YYYY-MM-DD
  • messages: массив объектов, каждый с полями
      – message_id (int)
      – date (string)
      – from_id (string)
      – text (string)
      – reply_to_message_id (int или null)

Правила выделения:
1. Вопросы — сообщения, у которых, from_id НЕ входит в список кураторов, и текст заканчивается знаком «?» или явно является вопросом.
2. Ответы — сообщения, у которых ответ может сопоставлятся по смыслу с вопросом и точно отвечает на него, и from_id входит в список кураторов:
   817658884, 596949549, 231422119, 47473273, 203511708, 444422071

Ваша задача:
— Найти все истинные пары «вопрос → ответ».
— Для каждой пары вернуть ровно один объект с полями:
    question_text — текст вопроса
    answer_text   — текст ответа
    topic         — краткая тема (1–5 слов), описывающая суть пары

Используйте список EXISTING_TOPICS. Если существующие темы не подходят, добавьте новые в массив new_topics.

Верните строго JSON в формате:
{
  "qa_pairs": [
    {
      "question_text": "...",
      "answer_text": "...",
      "topic": "..."
    },
    …
  ],
  "new_topics": [ … ]
}
Без какого-либо дополнительного текста или markdown-оформления.
"""

MAX_PROMPT_TOKENS = 6500        # небольшой запас

def split_by_tokens(msgs, model, max_tokens=MAX_PROMPT_TOKENS):
    """Разбивает список сообщений так, чтобы итоговая JSON-строка + промпт
       гарантированно помещалась < max_tokens."""
    batches, current, tokens = [], [], 0
    for m in msgs:
        t = len(model.tokenize(json.dumps(m, ensure_ascii=False)))
        if tokens + t > max_tokens and current:
            batches.append(current)
            current, tokens = [m], t
        else:
            current.append(m)
            tokens += t
    if current:
        batches.append(current)
    return batches


# Function: process one batch of messages
def process_batch(day, batch, existing_topics, idx):
    prompt = SYSTEM_PROMPT.replace("EXISTING_TOPICS", ", ".join(existing_topics) or "(none)")
    for attempt in range(1, MAX_RETRIES+1):
        thread = assistant = None
        try:
            thread    = sdk.threads.create(ttl_days=1, expiration_policy="static")
            assistant = sdk.assistants.create(model, ttl_days=1, expiration_policy="since_last_active")
            assistant.update(instruction=prompt)

            payload = json.dumps({"day": day, "messages": batch}, ensure_ascii=False)
            thread.write(payload)
            run = assistant.run(thread)
            res = run.wait()
            raw = (res.text or "").strip()
            # strip markdown fences if any
            if raw.startswith("```") and raw.endswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]).strip()
            data = json.loads(raw)
            thread.delete(); assistant.delete()
            return data.get("qa_pairs", []), data.get("new_topics", [])
        except Exception as e:
            logger.warning(f"Batch {day}-{idx} attempt {attempt} failed: {e}")
            if assistant:
                try: assistant.delete()
                except: pass
            if thread:
                try: thread.delete()
                except: pass
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SEC * (2 ** (attempt-1)))
            else:
                logger.error(f"Batch {day}-{idx} failed after {MAX_RETRIES} attempts")
                return [], []

# Function: extract QA pairs for one day
def extract_for_day(day, msgs, existing_topics):
    batches = split_by_tokens(msgs, model)
    all_qas = []
    new_topics = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(process_batch, day, b, existing_topics, i+1): i+1 for i,b in enumerate(batches)}
        for f in tqdm(as_completed(futures), total=len(batches), desc=f"Day {day}"):
            qas, nts = f.result()
            all_qas.extend(qas)
            for t in nts:
                if t and t not in existing_topics and t not in new_topics:
                    new_topics.append(t)
    return all_qas, new_topics

# Main pipeline
def main():
    # load all messages
    day_map = {}
    for file in sorted(ORIGINAL_DIR.glob("*.json")):
        raw = json.loads(file.read_text(encoding='utf-8')).get("messages", [])
        for m in raw:
            if m.get("type") != "message": continue
            day = m["date"][:10]
            txt = m["text"]
            if isinstance(txt, list):
                parts=[]
                for ent in txt:
                    parts.append(ent.get("text","") if isinstance(ent, dict) else ent)
                txt = "".join(parts)
            day_map.setdefault(day, []).append({
                "message_id": m["id"],
                "date": m["date"],
                "from_id": m.get("from_id"),
                "text": txt,
                "reply_to_message_id": m.get("reply_to_message_id")
            })
    days = sorted(day_map.keys())
    total_days = len(days)
    if total_days==0:
        logger.error("No message days found in %s", ORIGINAL_DIR)
        return
    
    # output accumulator per topic
    topic_accum = {}
    # process each day
    for idx, day in enumerate(days,1):
        pct = idx/total_days*100
        print(f"Processing day {idx}/{total_days} ({pct:.1f}%) — {day}")
        if day in state["processed_days"]:
            continue
        msgs = day_map[day]
        qas, new_ts = extract_for_day(day, msgs, topics)
        # update topics
        if new_ts:
            topics.extend(new_ts)
            TOPICS_FILE.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding='utf-8')
        # write per-topic JSON immediately for this day
        import json as _json
        for qa in qas:
            topic = qa.get("topic")
            if not topic:
                continue
            key = topic.replace(" ", "_")
            out_path = OUTPUT_DIR / f"{key}.json"
            # load existing if any
            if out_path.exists():
                existing = _json.loads(out_path.read_text(encoding='utf-8')).get("qa_pairs", [])
            else:
                existing = []
            existing.append({
                "question_text": qa["question_text"],
                "answer_text": qa["answer_text"],
                "topic": qa["topic"]
            })
            out_path.write_text(_json.dumps({"qa_pairs": existing}, ensure_ascii=False, indent=2), encoding='utf-8')
            logger.info(f"After day {day}, wrote {len(existing)} QA(s) to {out_path.name}")
        # checkpoint
        state["processed_days"].append(day)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        state["processed_days"].append(day)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    # write output files
    for t, items in topic_accum.items():
        out=OUTPUT_DIR/ f"{t}.json"
        out.write_text(json.dumps({"qa_pairs": items}, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(f"Wrote {out.name}: {len(items)} QA pairs")

if __name__ == "__main__":
    main()
