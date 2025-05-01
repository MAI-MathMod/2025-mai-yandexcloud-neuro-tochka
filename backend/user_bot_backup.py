import os
import asyncio
import logging
import threading
import signal
import requests
import aiohttp
import json
import csv
import io
import re
import time
import base64
import sqlite3
from datetime import datetime
import aiosqlite
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
)
from yandex_cloud_ml_sdk import YCloudML

# ─── Configuration & Logging ─────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="bot.log",
)
logger = logging.getLogger(__name__)


# Telegram & Admin-Bot endpoints
USER_BOT_TOKEN = os.getenv("USER_BOT_TOKEN")
ADMIN_BOT_TICKETS_URL = os.getenv(
    "ADMIN_BOT_TICKETS_URL", "http://admin-bot:8096/tickets"
)
ADMIN_BOT_MESSAGE_URL = os.getenv(
    "ADMIN_BOT_MESSAGE_URL", "http://admin-bot:8096/message"
)

# Yandex Cloud ML SDK creds
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_AUTH_TOKEN = os.getenv("YC_AUTH_TOKEN")
YC_MODEL_NAME = os.getenv("YC_MODEL_NAME", "yandexgpt")
ASSISTANT_NAME_RU = os.getenv("ASSISTANT_NAME_RU")
ASSISTANT_NAME_EN = os.getenv("ASSISTANT_NAME_EN")
ASSISTANT_NAME_ZH = os.getenv("ASSISTANT_NAME_ZH")
ASSISTANT_NAME_SUM = os.getenv("ASSISTANT_NAME2", "mai_thread_summarizer1")
INDEX_NAME_RU = os.getenv("INDEX_NAME_RU")
INDEX_NAME_EN = os.getenv("INDEX_NAME_EN")
INDEX_NAME_ZH = os.getenv("INDEX_NAME_ZH")

# Debug echo‐mode toggle
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
print(DEBUG_MODE)

# Google Sheets config
GOOGLE_SHEETS_URL = os.getenv(
    "GOOGLE_SHEETS_URL",
    "https://docs.google.com/spreadsheets/d/1RSmWjkxHnZ3GnWuRia5NkMGE9MTHROtWu0Cri5LXLBI/edit?usp=sharing",
)

# Validate required
if not USER_BOT_TOKEN:
    logger.error("Missing USER_BOT_TOKEN in .env")
    exit(1)
if not DEBUG_MODE and (not YC_FOLDER_ID or not YC_AUTH_TOKEN):
    logger.error(
        "Missing YC_FOLDER_ID or YC_AUTH_TOKEN in .env (unless DEBUG_MODE=true)"
    )
    exit(1)

# Initialize Yandex SDK if not in debug
sdk = None
if not DEBUG_MODE:
    sdk = YCloudML(folder_id=YC_FOLDER_ID, auth=YC_AUTH_TOKEN)
    sdk.setup_default_logging(log_level="DEBUG")


def clean_html(text: str) -> str:
    """
    Remove all <...> tags from a string and return clean text.
    """
    return re.sub(r"<[^>]+>", "", text)


def perform_yandex_search(
    query_body: dict, poll_interval: int = 5, timeout: int = 600
) -> str:
    """
    Perform an asynchronous web search using Yandex Search API and retrieve the raw results.

    This function submits a search query to the Yandex Search API's asynchronous endpoint,
    polls for completion of the search operation at specified intervals, and retrieves
    the Base64-encoded response once complete. The response is then decoded and returned
    as a UTF-8 string.

    The function implements a long-polling pattern with timeout protection to handle
    the asynchronous nature of the Yandex Search API. It uses bearer token authentication
    through the YC_AUTH_TOKEN environment variable.

    Parameters:
        query_body (dict): The search query and parameters formatted as a JSON-serializable
                          dictionary according to Yandex Search API specifications.
        poll_interval (int, optional): Time in seconds between polling requests to check
                                      operation status. Default is 5 seconds.
        timeout (int, optional): Maximum time in seconds to wait for the search operation
                               to complete before raising a TimeoutError. Default is 600 seconds.

    Returns:
        str: The decoded search results as a UTF-8 string.

    Raises:
        requests.exceptions.HTTPError: If any API request fails.
        TimeoutError: If the search operation does not complete within the specified timeout.
        Exception: For other potential errors in the requests or data processing.

    API Endpoints Used:
        - POST https://searchapi.api.cloud.yandex.net/v2/web/searchAsync - To initiate search
        - GET https://operation.api.cloud.yandex.net/operations/{id} - To check status

    Note:
        This function requires the YC_AUTH_TOKEN environment variable to be set with
        a valid Yandex Cloud API token with appropriate permissions.
    """
    headers = {
        "Authorization": f"Bearer {YC_AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    # Launch async search
    resp = requests.post(
        "https://searchapi.api.cloud.yandex.net/v2/web/searchAsync",
        headers=headers,
        json=query_body,
    )
    resp.raise_for_status()
    operation_id = resp.json().get("id")

    # Poll for completion
    op_url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
    start = time.time()
    while True:
        op_resp = requests.get(
            op_url, headers={"Authorization": f"Bearer {YC_AUTH_TOKEN}"}
        )
        op_resp.raise_for_status()
        op_data = op_resp.json()
        if op_data.get("done"):
            break
        if time.time() - start > timeout:
            raise TimeoutError("Yandex search operation timed out.")
        time.sleep(poll_interval)

    # Extract and decode Base64 rawData
    raw_base64 = op_data["response"]["rawData"]
    decoded = base64.b64decode(raw_base64).decode("utf-8", errors="ignore")
    return decoded


def call_yandexgpt(search_need: str, content: str) -> str:
    """
    Call Yandex GPT model with the search query and the search result content.
    Returns the structured answer text.
    """
    prompt = (
        f"Запрос для поиска: {search_need}\n"
        f"Собранные данные:\n{content}\n"
        "Пожалуйста, обработай данные и выдай структурированный ответ на вопрос."
    )
    model = sdk.models.completions("yandexgpt", model_version="rc")
    completion = model.run(prompt)
    return completion.text


def process_request_args(request_json: dict) -> str:
    """
    Main entrypoint function.
    Expects JSON with keys:
      - func_name: str
      - search_need: str

    Returns a dict with func_name and generated answer.
    """
    func_name = "SearchAndSumArgs"
    search_need = request_json["search_need"]
    if not func_name or not search_need:
        raise ValueError("Request JSON must include 'func_name' and 'search_need'.")

    # Build Yandex Search API body
    query_body = {
        "query": {
            "searchType": "SEARCH_TYPE_RU",
            "queryText": search_need,
            "familyMode": "FAMILY_MODE_MODERATE",
            "page": 0,
            "fixTypoMode": "FIX_TYPO_MODE_ON",
        },
        "sortSpec": {
            "sortMode": "SORT_MODE_BY_RELEVANCE",
            "sortOrder": "SORT_ORDER_DESC",
        },
        "groupSpec": {
            "groupMode": "GROUP_MODE_DEEP",
            "groupsOnPage": 7,
            "docsInGroup": 1,
        },
        "maxPassages": 4,
        "region": "213",
        "l10N": "LOCALIZATION_RU",
        "folderId": YC_FOLDER_ID,
        "responseFormat": "FORMAT_XML",
        "userAgent": "Mozilla/5.0",
    }

    # 1. Perform search and get raw HTML/XML
    raw_response = perform_yandex_search(query_body)
    # 2. Clean HTML/XML tags
    clean_text = clean_html(raw_response)
    # 3. Call YandexGPT for structured answer
    answer = call_yandexgpt(search_need, clean_text)

    return answer


def get_or_create_index(index_name):
    """
    Find existing search index or create a new one.

    This function searches through the available Yandex Cloud search indexes to
    find one matching the configured INDEX_NAME. It returns the found index, or
    creates a new one if no matching index is found.

    Returns:
        object: The search index object that matches INDEX_NAME
    """
    z = []
    for idx in sdk.search_indexes.list():
        if idx.name == index_name:
            z.append(idx)
    if z:
        print(f"✔️ Индекс найден: {z[-1].id}")
        return z[-1]
    print("◆ Индекс не найден — создаём заново…")


def get_or_create_assistant(index, assistant_name):
    """
    Find existing assistant or create a new one with the specified name.

    This function searches through the available Yandex Cloud assistants to find
    one matching the specified assistant_name. It returns the found assistant, or
    potentially creates a new one if no matching assistant is found.

    Args:
        index: The search index to associate with the assistant
        assistant_name (str, optional): Name of the assistant to find.
                                        Defaults to ASSISTANT_NAME.

    Returns:
        object: The assistant object that matches the requested name
    """
    z = []
    for a in sdk.assistants.list():
        if a.name == assistant_name:
            z.append(a)
    if z:
        print(f"✔️ Ассистент {assistant_name} найден: {z[-1].id}")
        return z[-1]
    print(f"◆ Ассистент {assistant_name} не найден — создаём заново…")


# ─── In-Memory State ─────────────────────────────────────────────────────────

dialog_history = {}  # user_id → [ {"role","content"}, … ]
user_threads = {}  # user_id → YandexLLM thread
pending_tickets = {}  # user_id → ticket_id (escalated but not claimed)
pending_ticket_to_user = {}  # ticket_id → user_id
tickets_to_user = {}  # ticket_id → user_id (claimed)
user_states = {}  # user_id → {"state": "waiting_for_score", "data": {...}}
user_profiles = {}  # user_id → {"name": "", "interests": "", "courses": ""}
user_languages = {}  # user_id → "en"|"ru"|"zh" (English, Russian, Chinese)

# Will hold our asyncio event loop
polling_loop = None
yandex_assistant_ru = None
yandex_assistant_en = None
yandex_assistant_zh = None
yandex_summarizer = None

# ─── Flask & Aiogram Setup ─────────────────────────────────────────────────

app = Flask(__name__)
bot = Bot(token=USER_BOT_TOKEN)
dp = Dispatcher()

# ─── Helpers ────────────────────────────────────────────────────────────────


async def build_user_context_message(user_id):
    """
    Build a context message containing user profile information for the LLM.

    This function retrieves a user's profile from the database and constructs a
    formatted message that provides context about the user to the language model.
    The context can include the user's name, a profile summary (AI-generated),
    interests, and courses they're taking.

    Args:
        user_id (int): The Telegram user ID to retrieve profile information for

    Returns:
        str or None: A formatted context message if user profile exists and contains
                     relevant information, otherwise None

    The function prioritizes using the AI-generated profile summary if available,
    but will fall back to using raw interests and courses if no summary exists.
    """
    try:
        user_profile = await get_user_profile(user_id)
        if not user_profile:
            return None

        context_parts = []

        if user_profile.get("name"):
            context_parts.append(f"Name: {user_profile.get('name')}")

        if user_profile.get("profile_summary"):
            context_parts.append(
                f"Profile summary: {user_profile.get('profile_summary')}"
            )
        elif user_profile.get("interests") or user_profile.get("courses"):
            if user_profile.get("interests"):
                context_parts.append(f"Interests: {user_profile.get('interests')}")
            if user_profile.get("courses"):
                context_parts.append(f"Courses: {user_profile.get('courses')}")

        if not context_parts:
            return None

        context_message = "USER CONTEXT:\n" + "\n".join(context_parts)
        context_message += (
            "\n\nPlease personalize your responses based on this user context."
        )
        return context_message
    except Exception as e:
        logger.exception(f"[ERROR] Error building context message for user {user_id}")
        return None


async def call_yc(user_id: int, text: str) -> str:
    """
    Process user messages through the Yandex Cloud LLM with threads and assistants.

    This function is the core interaction point with the Yandex Cloud ML model. It:
    1. Manages thread creation and message history
    2. Adds user profile context to enhance personalization
    3. Implements retry logic for resilience against API errors
    4. Handles thread locking issues by creating new threads when needed
    5. Processes responses and detects escalation signals

    Args:
        user_id (int): The Telegram user ID sending the message
        text (str): The message content from the user

    Returns:
        tuple: (response_text, escalate_flag) where:
            - response_text (str): The model's response text
            - escalate_flag (bool): True if the model indicated this should be escalated to a human

    The function implements extensive error handling and logging, capturing the complete
    context, input and response for debugging purposes. In debug mode, it will simply
    echo back the input text.
    """
    lang = user_languages.get(user_id, "ru")
    escalate_flag = False
    if DEBUG_MODE:
        logger.info(f"[REQUEST] Debug mode echo for user {user_id}: '{text}'")
        return text

    thread = user_threads.get(user_id)
    if not thread:
        thread = sdk.threads.create(ttl_days=7, expiration_policy="static")
        user_threads[user_id] = thread

        context_message = await build_user_context_message(user_id)
        if context_message:
            try:
                try:
                    thread.write(context_message)
                    logger.info(
                        f"[CONTEXT] Added profile context for user {user_id}:\n{context_message}"
                    )
                except Exception:
                    logger.warning(
                        f"[CONTEXT] Could not add system context for user {user_id}, using user message instead"
                    )
                    thread.write(
                        f"SYSTEM CONTEXT (please remember this): {context_message}"
                    )
                    logger.info(
                        f"[CONTEXT] Added as user message for user {user_id}:\nSYSTEM CONTEXT (please remember this): {context_message}"
                    )
            except Exception as e:
                logger.exception(
                    f"[ERROR] Error adding profile context for user {user_id}"
                )

    max_retries = 3
    retry_delay = 2  # seconds

    # Add instruction to focus on the current question

    focused_text_ru = f"НОВЫЙ ВОПРОС (отвечай только на этот вопрос, не повторяй предыдущие ответы): {text}, ОБЯЗАТЕЛЬНО ПРИМЕНИ ПОИСК В ИНТЕРНЕТЕ"
    focused_text_en = f"NEW QUESTION (answer only this question, do not repeat previous answers): {text}, ALWAYS USE SEARCH IN THE INTERNET"
    focused_text_zh = (
        f"新问题（仅回答此问题，不要重复之前的答案）：{text}，请在互联网上搜索"
    )
    if lang == "ru":
        focused_text = focused_text_ru
    elif lang == "en":
        focused_text = focused_text_en
    elif lang == "zh":
        focused_text = focused_text_zh

    for attempt in range(max_retries):
        try:
            # Write user message to thread with the focus instruction
            thread.write(focused_text)
            logger.info(f"[REQUEST] User {user_id}: '{text}'")
            break  # Success, exit the loop
        except Exception as e:
            if "is locked by run" in str(e) and attempt < max_retries - 1:
                logger.warning(
                    f"Thread for user {user_id} is locked, retry {attempt+1}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue

            if attempt == max_retries - 1:
                logger.warning(
                    f"Creating new thread for user {user_id} after failed attempts"
                )
                try:
                    thread = sdk.threads.create(ttl_days=7, expiration_policy="static")
                    user_threads[user_id] = thread

                    context_message = await build_user_context_message(user_id)
                    if context_message:
                        try:
                            thread.write(context_message)
                            logger.info(
                                f"[CONTEXT] Added profile context to new thread for user {user_id}:\n{context_message}"
                            )
                        except Exception as profile_error:
                            try:
                                thread.write(
                                    f"SYSTEM CONTEXT (please remember this): {context_message}"
                                )
                                logger.info(
                                    f"[CONTEXT] Added context as user message for user {user_id}:\nSYSTEM CONTEXT (please remember this): {context_message}"
                                )
                            except Exception:
                                logger.exception(
                                    f"[ERROR] Error adding profile context to new thread"
                                )

                    # Write the original message with focus instruction to new thread
                    thread.write(focused_text)
                    logger.info(f"[REQUEST] User {user_id}: '{text}' (new thread)")
                except Exception as new_thread_error:
                    logger.exception(f"Failed to create new thread for user {user_id}")
                    raise new_thread_error
            else:
                logger.exception(f"Error writing message to thread for user {user_id}")
                raise e

    try:
        try:
            messages = thread.list_messages()
            if messages:
                thread_history = "\n---\n".join(
                    [
                        f"[{m.role if hasattr(m, 'role') else 'user'}]: {m.content}"
                        for m in messages
                    ]
                )
                logger.info(
                    f"[THREAD] Complete history for user {user_id} before running assistant:\n{thread_history}"
                )
        except Exception:
            logger.warning(
                f"[THREAD] Could not retrieve full thread history for user {user_id}"
            )
    except Exception:
        pass  # Skip thread history logging if it fails

    try:
        if lang == "ru":
            run = yandex_assistant_ru.run(thread)
            res = run.wait()
        elif lang == "en":
            run = yandex_assistant_en.run(thread)
            res = run.wait()
        elif lang == "zh":
            run = yandex_assistant_zh.run(thread)
            res = run.wait()

        response = (res.text or "").strip()

        # Log the full model response
        logger.info(f"[RESPONSE] For user {user_id}, model returned:\n{response}")

        if res.tool_calls:
            result = []
            for i in res.tool_calls:
                if i.function.name == "EscalateTicket":
                    escalate_flag = True
                    logger.info(
                        f"[ESCALATE] Model triggered escalation for user {user_id}"
                    )
                    break
                elif i.function.name == "SearchAndSumArgs":
                    x = process_request_args(i.function.arguments)
                    result.append({"name": i.function.name, "content": x})
                    logger.info(
                        f"[SearchAndSumArgs] Model triggered search for user {user_id}\n"
                        + f"[SearchAndSumArgs] {run}"
                    )
                    print(result)
            run.submit_tool_results(result)
            time.sleep(3)
            res = run.wait()
            response = (res.text or "").strip()

        if res.citations:
            logger.info(
                f"[CITATIONS] Response for user {user_id} includes citations: {res.citations}"
            )

        return response, escalate_flag
    except Exception as e:
        logger.exception(f"[ERROR] Error running assistant for user {user_id}")
        return (
            "I'm experiencing technical difficulties right now. You may want to try again later or use the /escalate command to reach a human admin.",
            True,
        )


async def summarize_profile_async(user_id: int, profile_text: str):
    """
    Asynchronously generate an AI summary of a user's profile using a dedicated LLM.

    This function uses a separate YandexGPT model specifically optimized for profile
    summarization. It creates a temporary thread, sends the profile data, and extracts
    a concise summary highlighting key attributes of the user. The summary is then
    stored in the database for future use in personalizing conversations.

    Args:
        user_id (int): The Telegram user ID whose profile is being summarized
        profile_text (str): The raw profile text containing name, interests, and courses

    Returns:
        None: This function operates asynchronously and doesn't return a value

    The function includes built-in retry logic for database operations and will
    attempt multiple approaches to save the summary if the primary method fails.
    If running in DEBUG mode or if the summarizer assistant is not available,
    this function will log a message and exit without performing summarization.
    """
    if DEBUG_MODE or not yandex_summarizer:
        logger.info(
            f"[SUMMARY] Skipping summarization for user {user_id} - DEBUG or no summarizer"
        )
        return

    await asyncio.sleep(1.5)

    try:
        summary_thread = sdk.threads.create(ttl_days=1, expiration_policy="static")

        summary_prompt = f"Please summarize this user profile concisely highlighting key attributes, interests, and educational goals:\n\n{profile_text}"

        summary_thread.write(summary_prompt)
        logger.info(
            f"[SUMMARY] Requesting summary for user {user_id} with prompt:\n{summary_prompt}"
        )

        run = yandex_summarizer.run(summary_thread)
        res = run.wait()

        summary = (res.text or "").strip()

        if summary:
            logger.info(f"[SUMMARY] Generated summary for user {user_id}:\n{summary}")
            max_retries = 3
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    if retry_count > 0:
                        await asyncio.sleep(1.0 * retry_count)

                    update_success = await update_profile_summary(user_id, summary)
                    if update_success:
                        logger.info(
                            f"[SUMMARY] Saved profile summary for user {user_id}"
                        )
                        success = True
                    else:
                        logger.warning(
                            f"[SUMMARY] Failed to save summary for user {user_id}, attempt {retry_count+1}"
                        )
                        retry_count += 1
                except Exception as e:
                    logger.exception(
                        f"[ERROR] Error saving summary for user {user_id}, attempt {retry_count+1}"
                    )
                    retry_count += 1

            if not success:
                try:
                    db_profile = await get_user_profile(user_id)
                    if db_profile:
                        success = await save_user_profile(
                            user_id,
                            db_profile.get("name", ""),
                            db_profile.get("interests", ""),
                            db_profile.get("courses", ""),
                            profile_summary=summary,
                        )
                        if success:
                            logger.info(
                                f"[SUMMARY] Saved summary with full profile update for user {user_id}"
                            )
                except Exception as e:
                    logger.exception(
                        f"[ERROR] Failed in fallback save of summary for user {user_id}"
                    )

        try:
            summary_thread.delete()
        except Exception:
            pass

    except Exception as e:
        logger.exception(
            f"[ERROR] Failed to generate profile summary for user {user_id}"
        )


async def update_profile_summary(user_id, profile_summary):
    """
    Update only the profile summary field in a user's profile.

    This function specifically updates the AI-generated summary of a user's profile
    without modifying other profile fields. It includes fallback logic to handle
    database schema changes or missing columns.

    Args:
        user_id (int): The Telegram user ID to update the summary for
        profile_summary (str): The new AI-generated profile summary

    Returns:
        bool: True if the update succeeded, False otherwise

    If the direct update fails (e.g., due to a missing column), the function will
    attempt a full profile update as a fallback.
    """
    try:
        async with get_db_connection() as db:
            current_time = datetime.now().isoformat()

            try:
                await db.execute(
                    """
                UPDATE user_profiles 
                SET profile_summary = ?, updated_at = ?
                WHERE user_id = ?
                """,
                    (profile_summary, current_time, user_id),
                )
                await db.commit()
                logger.info(f"[SUMMARY] Updated profile summary for user {user_id}")
                return True
            except sqlite3.OperationalError as e:
                logger.warning(f"[ERROR] Column error when updating summary: {str(e)}")

                # Get existing profile data
                async with db.execute(
                    "SELECT name, interests, courses FROM user_profiles WHERE user_id = ?",
                    (user_id,),
                ) as cursor:
                    result = await cursor.fetchone()

                if result:
                    # Update the entire profile with the new summary
                    name, interests, courses = (
                        result["name"],
                        result["interests"],
                        result["courses"],
                    )
                    await db.execute(
                        """
                    UPDATE user_profiles 
                    SET name = ?, interests = ?, courses = ?, profile_summary = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                        (
                            name,
                            interests,
                            courses,
                            profile_summary,
                            current_time,
                            user_id,
                        ),
                    )
                    await db.commit()
                    logger.info(
                        f"[SUMMARY] Updated full profile with summary for user {user_id}"
                    )
                    return True
                return False
    except Exception as e:
        logger.exception(f"[ERROR] Database error updating summary for user {user_id}")
        return False


async def escalate_to_admin(user_id: int, dialog: list) -> str:
    """
    Escalate a user conversation to a human admin by creating a ticket.

    This function sends the user's dialog history to the Admin-Bot service
    to create a support ticket. It registers the pending ticket in memory
    for tracking until an admin claims it.

    Args:
        user_id (int): The Telegram user ID requesting escalation
        dialog (list): A list of dialog messages with 'role' and 'content' keys

    Returns:
        str: The ticket ID created for this escalation

    Raises:
        Exception: If the Admin-Bot service is unreachable or returns an error
    """
    payload = {"user_id": user_id, "dialog": dialog}
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        resp = await sess.post(ADMIN_BOT_TICKETS_URL, json=payload)
        resp.raise_for_status()
        data = await resp.json()
    ticket_id = data["ticket_id"]
    pending_tickets[user_id] = ticket_id
    pending_ticket_to_user[ticket_id] = user_id
    logger.info(f"[TICKET] Created ticket {ticket_id} for user {user_id}")
    return ticket_id


async def forward_to_admin(ticket_id: str, text: str):
    """
    Forward a user message to an admin after a ticket has been claimed.

    This function sends a new user message to the Admin-Bot service for
    an existing ticket that has already been claimed by an admin.

    Args:
        ticket_id (str): The ticket ID to forward the message to
        text (str): The message text to forward

    Returns:
        None
    """
    payload = {"ticket_id": ticket_id, "text": text}
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(ADMIN_BOT_MESSAGE_URL, json=payload)
            logger.info(f"[ADMIN] Forwarded message to ticket {ticket_id}")
            if resp.status != 200:
                body = await resp.text()
                logger.warning(
                    f"[ERROR] Admin-Bot rejected forward ({resp.status}): {body}"
                )
    except Exception:
        logger.exception(f"[ERROR] Failed to forward to admin for ticket {ticket_id}")


async def fetch_scores():
    """
    Fetch exam scores from a Google Sheet using the public CSV export feature.

    This function retrieves student exam scores from a Google Spreadsheet,
    processes the CSV data, and returns a structured representation of the scores.
    The scores are sorted in descending order by value.

    Returns:
        dict: A dictionary containing either:
            - {"items": [{"name": "СНИЛС: xxx", "score": 123}, ...]} for successful requests
            - {"error": "error message"} if there was an issue fetching or processing the data

    The function extracts the sheet ID from the URL defined in GOOGLE_SHEETS_URL
    environment variable and constructs a CSV export URL to fetch the data.
    """
    try:
        # Extract sheet ID from the URL
        sheet_id = GOOGLE_SHEETS_URL.split("/d/")[1].split("/edit")[0]

        # Construct CSV export URL
        csv_export_url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        )

        # Fetch the CSV data
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(csv_export_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch Google Sheet: {response.status}")
                    return {"error": "Unable to access the Google Sheet at this time"}

                # Read CSV content
                content = await response.text()

                # Parse CSV
                items = []
                reader = csv.DictReader(io.StringIO(content))
                for row in reader:
                    # Check if this row has the required data (СНИЛС and Баллы fields)
                    if row.get("Снилс") and row.get("Баллы"):
                        try:
                            # Convert score to integer for sorting
                            score = int(row["Баллы"])
                            items.append(
                                {"name": f"СНИЛС: {row['Снилс']}", "score": score}
                            )
                        except ValueError:
                            # Skip rows with non-numeric scores
                            continue

                # Sort items by score in descending order
                items.sort(key=lambda x: x["score"], reverse=True)

                return {"items": items}
    except Exception as e:
        logger.exception("Error fetching scores from Google Sheet")
        return {"error": "Unable to access scores at this time"}


# ─── Database Setup ─────────────────────────────────────────────────────────

# Global database lock to prevent concurrent writes
db_lock = asyncio.Lock()
db_timeout = 30.0  # Timeout for database operations
db_pool = {}  # Connection pool


class DatabaseConnectionManager:
    """
    A connection pool manager for SQLite database access in an asynchronous environment.

    This class provides a task-based connection pooling mechanism to manage SQLite
    database connections efficiently while ensuring thread safety in an async context.
    It maintains a dictionary of database connections keyed by task ID to allow
    different async tasks to use separate connections without interference.

    The manager implements a connection-per-task pattern, where each async task
    receives its own dedicated database connection. This approach prevents connection
    sharing issues in the async environment while allowing connection reuse within
    the same task.

    Features:
    - Task-local connection management to prevent concurrency issues
    - Automatic connection creation on first access by a task
    - Connection pooling to avoid excessive connection creation/destruction
    - Thread-safe access through asyncio locks
    - Complete connection cleanup capability

    The connection manager is designed to work with the aiosqlite library and
    SQLite databases. It's particularly useful in a Telegram bot application where
    multiple async handlers might need database access concurrently.

    Usage Example:
        db_manager = DatabaseConnectionManager()

        async def some_task():
            conn = await db_manager.get_connection()
            await conn.execute(...)
            await conn.commit()

    Note: Most application code should use the get_db_connection() context manager
    instead of accessing this class directly.
    """

    def __init__(self):
        self.lock = asyncio.Lock()
        self.connections = {}

    async def get_connection(self):
        """Get a connection from the pool or create a new one"""
        # Use the current task as a key to track connections
        task_id = id(asyncio.current_task())

        async with self.lock:
            if task_id not in self.connections:
                # Create a new connection
                conn = await aiosqlite.connect("user_profiles.db", timeout=db_timeout)
                conn.row_factory = aiosqlite.Row
                self.connections[task_id] = conn

            return self.connections[task_id]

    async def close_all(self):
        """Close all connections in the pool"""
        async with self.lock:
            for conn in self.connections.values():
                try:
                    await conn.close()
                except Exception:
                    pass
            self.connections.clear()


# Create a connection manager
db_manager = DatabaseConnectionManager()


@asynccontextmanager
async def get_db_connection():
    """
    Get a database connection with proper locking and error handling.

    This function serves as an async context manager that provides thread-safe
    access to the SQLite database. It uses a global lock to prevent concurrent
    writes that could corrupt the database.

    Usage:
        async with get_db_connection() as db:
            await db.execute(...)

    Returns:
        aiosqlite.Connection: A connection to the SQLite database

    The connection is managed by the connection pool and not closed when the
    context manager exits, as it may be reused for future operations.
    """
    async with db_lock:
        connection = None
        try:
            connection = await db_manager.get_connection()
            yield connection
        finally:
            pass


async def setup_database():
    """
    Initialize the SQLite database schema for user profiles.

    This function creates the necessary database tables if they don't exist
    and ensures all required columns are present. It will automatically
    add new columns (like profile_summary) to existing tables if needed.

    Returns:
        None
    """
    async with get_db_connection() as db:
        # Create the main table if it doesn't exist
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            interests TEXT,
            courses TEXT,
            agent_response TEXT,
            profile_summary TEXT,
            language TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
        )

        # Check if profile_summary column exists, and add it if it doesn't
        try:
            # Try to select from profile_summary column
            await db.execute("SELECT profile_summary FROM user_profiles LIMIT 1")
        except sqlite3.OperationalError:
            # If column doesn't exist, add it
            logger.info("Adding profile_summary column to user_profiles table")
            await db.execute(
                "ALTER TABLE user_profiles ADD COLUMN profile_summary TEXT"
            )

        # Check if language column exists, and add it if it doesn't
        try:
            # Try to select from language column
            await db.execute("SELECT language FROM user_profiles LIMIT 1")
        except sqlite3.OperationalError:
            # If column doesn't exist, add it
            logger.info("Adding language column to user_profiles table")
            await db.execute("ALTER TABLE user_profiles ADD COLUMN language TEXT")

        await db.commit()
    logger.info("Database initialized successfully")


async def save_user_profile(
    user_id,
    name,
    interests,
    courses,
    agent_response="",
    profile_summary=None,
    language=None,
):
    """
    Save or update a user's profile in the database.

    This function stores the user's profile information in the SQLite database,
    either by updating an existing profile or creating a new one. It records
    both the profile data and timestamps for creation/updates.

    Args:
        user_id (int): The Telegram user ID to save the profile for
        name (str): User's name
        interests (str): User's interests
        courses (str): User's courses or subjects
        agent_response (str, optional): Any agent/bot response to save with profile
        profile_summary (str, optional): AI-generated summary of the user profile
        language (str, optional): User's preferred language (en, ru, zh)

    Returns:
        bool: True if the operation succeeded, False otherwise

    The function handles both INSERT and UPDATE operations internally based on
    whether the user already exists in the database.
    """
    try:
        async with get_db_connection() as db:
            # Check if user already exists
            async with db.execute(
                "SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)
            ) as cursor:
                user_exists = await cursor.fetchone()

            current_time = datetime.now().isoformat()

            if user_exists:
                # Update existing user
                if profile_summary:
                    # Update with summary
                    await db.execute(
                        """
                    UPDATE user_profiles 
                    SET name = ?, interests = ?, courses = ?, agent_response = ?, profile_summary = ?, language = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                        (
                            name,
                            interests,
                            courses,
                            agent_response,
                            profile_summary,
                            language,
                            current_time,
                            user_id,
                        ),
                    )
                else:
                    # Update without changing summary
                    await db.execute(
                        """
                    UPDATE user_profiles 
                    SET name = ?, interests = ?, courses = ?, agent_response = ?, language = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                        (
                            name,
                            interests,
                            courses,
                            agent_response,
                            language,
                            current_time,
                            user_id,
                        ),
                    )
                logger.info(f"[PROFILE] Updated profile for user {user_id}")
            else:
                # Insert new user
                await db.execute(
                    """
                INSERT INTO user_profiles (user_id, name, interests, courses, agent_response, profile_summary, language, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        name,
                        interests,
                        courses,
                        agent_response,
                        profile_summary or "",
                        language,
                        current_time,
                        current_time,
                    ),
                )
                logger.info(f"[PROFILE] Created new profile for user {user_id}")

            await db.commit()
            return True
    except Exception as e:
        logger.exception(f"[ERROR] Database error saving profile for user {user_id}")
        return False


async def get_user_profile(user_id):
    """
    Retrieve a user's profile from the database.

    This function fetches the complete user profile information from the database,
    including name, interests, courses, and any AI-generated profile summary.

    Args:
        user_id (int): The Telegram user ID to retrieve the profile for

    Returns:
        dict or None: A dictionary containing the user's profile information if found,
                     otherwise None. Keys include 'name', 'interests', 'courses',
                     'agent_response', 'created_at', 'updated_at', 'profile_summary',
                     and 'language'.
    """
    try:
        async with get_db_connection() as db:
            async with db.execute(
                """
            SELECT name, interests, courses, agent_response, created_at, updated_at, profile_summary, language
            FROM user_profiles WHERE user_id = ?
            """,
                (user_id,),
            ) as cursor:
                result = await cursor.fetchone()

            if result:
                return {
                    "name": result["name"],
                    "interests": result["interests"],
                    "courses": result["courses"],
                    "agent_response": result["agent_response"],
                    "created_at": result["created_at"],
                    "updated_at": result["updated_at"],
                    "profile_summary": (
                        result["profile_summary"]
                        if "profile_summary" in result.keys()
                        else None
                    ),
                    "language": (
                        result["language"] if "language" in result.keys() else None
                    ),
                }
            return None
    except Exception as e:
        logger.exception(
            f"[ERROR] Database error retrieving profile for user {user_id}"
        )
        return None


# ─── Telegram: State Management for Conversations ───────────────────────────

# Simple state management for dialogs
user_states = {}  # user_id → {"state": "waiting_for_score", "data": {...}}

# ─── Telegram: Command Handlers ────────────────────────────────────────────


@dp.message(lambda message: message.text and message.text.strip().lower() == "/start")
async def handle_start(message: Message):
    """
    Handle the /start command to begin user profiling.

    This function first asks users to select their preferred language (Chinese, English,
    or Russian) using buttons. After language selection, it continues the regular user
    profiling flow to collect name, interests, and courses.

    Args:
        message (Message): The Telegram message containing the /start command

    Returns:
        None
    """
    uid = message.from_user.id

    # Initialize user profile
    user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Create language selection buttons
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇨🇳 Chinese", callback_data="lang_zh"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="🇷🇺 Russian", callback_data="lang_ru"),
            ],
        ]
    )

    # Set user state to waiting for language selection
    user_states[uid] = {"state": "waiting_for_language", "data": {}}

    await message.reply(
        "👋 Welcome! Please select your preferred language:\n\n"
        "欢迎！请选择您的首选语言：\n\n"
        "Добро пожаловать! Пожалуйста, выберите предпочитаемый язык:",
        reply_markup=keyboard,
    )


@dp.callback_query(lambda c: c.data.startswith("lang_"))
async def process_language_selection(callback_query: CallbackQuery):
    """
    Handle language selection from the buttons shown in the /start command.

    This function processes the user's language selection, stores it in both memory
    and database, and then continues with the regular profile creation flow by
    asking for the user's name.

    Args:
        callback_query (CallbackQuery): The callback query from button selection

    Returns:
        None
    """
    uid = callback_query.from_user.id
    language_code = callback_query.data.split("_")[1]  # Get "en", "ru", or "zh"

    # Store language preference in memory
    user_languages[uid] = language_code

    # Update user state to proceed with name collection
    user_states[uid] = {"state": "waiting_for_name", "data": {}}

    # Prepare welcome message in selected language
    if language_code == "zh":
        welcome_message = "谢谢！请告诉我您的名字："
    elif language_code == "ru":
        welcome_message = "Спасибо! Пожалуйста, скажите мне, как вас зовут:"
    else:  # Default to English
        welcome_message = "Thank you! Please tell me your name:"

    # Log language selection
    logger.info(f"[LANGUAGE] User {uid} selected language: {language_code}")

    # Update the message to remove buttons and show the next question
    await callback_query.message.edit_text(welcome_message)

    # Acknowledge the callback query
    await callback_query.answer()


@dp.message(lambda message: message.text and message.text.strip().lower() == "/help")
async def handle_help(message: Message):
    """
    Handle the /help command to display available features and commands.

    This function provides a comprehensive help message explaining the bot's
    capabilities, available commands, and purpose. If the user doesn't have
    a profile yet, it also offers a button to create one.

    Args:
        message (Message): The Telegram message containing the /help command

    Returns:
        None
    """
    lang = user_languages.get(message.from_user.id, "en")
    help_text_en = (
        "*📚 Exam Assistant Bot - Help Guide*\n\n"
        "*Available Commands:*\n"
        "• /start - Begin user profiling for personalized assistance\n"
        "• /profile - View your saved profile information\n"
        "• /updateprofile - Update your profile details\n"
        "• /viewscores - View current exam scores\n"
        "• /checkscore - Check your potential ranking with your score\n"
        "• /subscribe - Subscribe to exam results notifications\n"
        "• /escalate - Escalate to a human admin for assistance\n"
        "• /help - Show this help message\n\n"
        "🔍 *User Profiling*\n"
        "Creating a profile helps me provide personalized assistance tailored to your needs. "
        "Your profile information is analyzed by our AI to give you better recommendations and answers.\n\n"
        "You can also ask me any question about exams, courses, or educational topics!"
    )
    help_text_ru = (
        "*📚 Экзаменационный Ассистент - Справка*\n\n"
        "*Доступные команды:*\n"
        "• /start - Начать создание профиля для персонализированной помощи\n"
        "• /profile - Посмотреть сохраненную информацию о профиле\n"
        "• /updateprofile - Обновить данные профиля\n"
        "• /viewscores - Посмотреть текущие баллы экзаменов\n"
        "• /checkscore - Проверить потенциальный рейтинг с вашим баллом\n"
        "• /subscribe - Подписаться на уведомления о результатах экзаменов\n"
        "• /escalate - Эскалировать к администратору для помощи\n"
        "• /help - Показать это сообщение справки\n\n"
        "🔍 *Создание профиля*\n"
        "Создание профиля помогает мне предоставлять персонализированную помощь, адаптированную к вашим потребностям. "
    )
    help_text_zh = (
        "*📚 考试助手 - 帮助指南*\n\n"
        "*可用命令:*\n"
        "• /start - 开始用户配置文件以获取个性化帮助\n"
        "• /profile - 查看您的保存的个人资料信息\n"
        "• /updateprofile - 更新您的个人资料详细信息\n"
        "• /viewscores - 查看当前考试分数\n"
        "• /checkscore - 检查您分数的潜在排名\n"
        "• /subscribe - 订阅考试结果通知\n"
        "• /escalate - 升级到人类管理员以获取帮助\n"
        "• /help - 显示此帮助消息\n\n"
        "🔍 *用户配置文件*\n"
        "创建个人资料可以帮助我提供量身定制的个性化帮助。 "
    )
    match lang:
        case "zh":
            help_text = help_text_zh
        case "ru":
            help_text = help_text_ru
        case "en":
            help_text = help_text_en

    # Check if user has a profile
    uid = message.from_user.id
    has_profile = uid in user_profiles or await get_user_profile(uid) is not None

    # Add create profile button if they don't have one
    if not has_profile:
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📝 Create Profile", callback_data="create_profile"
                    )
                ]
            ]
        )
        await message.reply(help_text, parse_mode="Markdown", reply_markup=keyboard)
    else:
        await message.reply(help_text, parse_mode="Markdown")


@dp.message(
    lambda message: message.text and message.text.strip().lower() == "/viewscores"
)
async def handle_view_scores(message: Message):
    """
    Handle the /viewscores command to display current exam scores.

    This function fetches and displays the current exam scores from a Google Sheet.
    It presents the scores in a formatted message and provides a refresh button
    to allow users to get updated scores.

    Args:
        message (Message): The Telegram message containing the /viewscores command

    Returns:
        None
    """
    uid = message.from_user.id
    lang = user_languages.get(uid, "en")

    # Create a keyboard with button to refresh scores
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Refresh Scores", callback_data="refresh_scores"
                )
            ]
        ]
    )

    # Fetch scores
    scores = await fetch_scores()

    if "error" in scores:
        error_text = (
            f"❌ *Error:* {scores['error']}\n\nPlease try again later."
            if lang == "en"
            else (
                f"❌ *Ошибка:* {scores['error']}\n\nПожалуйста, попробуйте позже."
                if lang == "ru"
                else f"❌ *错误:* {scores['error']}\n\n请稍后再试。"
            )
        )
        await message.reply(
            error_text,
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
    else:
        # Format scores for display
        scores_text = (
            "*📊 当前分数:*\n\n"
            if lang == "zh"
            else (
                "*📊 Текущие баллы:*\n\n"
                if lang == "ru"
                else "*📊 Current Scores:*\n\n"
            )
        )
        for item in scores.get("items", []):
            scores_text += f"• {item['name']}: {item['score']}\n"

        if not scores.get("items", []):
            scores_text += (
                "_当前没有分数。_\n"
                if lang == "zh"
                else (
                    "_Текущие баллы недоступны в настоящее время._\n"
                    if lang == "ru"
                    else "_No scores available at this time._\n"
                )
            )

        await message.reply(scores_text, parse_mode="Markdown", reply_markup=keyboard)


@dp.message(
    lambda message: message.text and message.text.strip().lower() == "/subscribe"
)
async def handle_subscribe(message: Message):
    """Handle the /subscribe command for exam results"""
    uid = message.from_user.id
    lang = user_languages.get(uid, "en")

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Subscribe", callback_data="subscribe_yes"
                ),
                InlineKeyboardButton(text="❌ Cancel", callback_data="subscribe_no"),
            ]
        ]
    )
    reply_text = (
        "*📣 Exam Results Notification Service*\n\n"
        "Would you like to subscribe to exam results notifications?\n\n"
        "You will be notified as soon as new results become available."
        if lang == "en"
        else (
            "*📣 考试结果通知服务*\n\n"
            "您想订阅考试结果通知吗？\n\n"
            "您将尽快收到新结果的通知。"
            if lang == "zh"
            else "*📣 Уведомление о результатах экзаменов*\n\n"
            "Вы хотите подписаться на уведомления о результатах экзаменов?\n\n"
            "Вы будете уведомлены, как только станут доступны новые результаты."
        )
    )

    await message.reply(reply_text, parse_mode="Markdown", reply_markup=keyboard)


@dp.message(
    lambda message: message.text and message.text.strip().lower() == "/checkscore"
)
async def handle_check_score(message: Message):
    """
    Handle the /checkscore command to check potential ranking position.

    This function allows users to check where they would rank in the current
    exam standings based on a score. It presents a set of score range buttons
    for quick selection as well as an option to enter a custom score.

    Args:
        message (Message): The Telegram message containing the /checkscore command

    Returns:
        None
    """
    uid = message.from_user.id
    logger.info(f"[SCORE] User {uid} requested score check")
    lang = user_languages.get(uid, "en")

    # Create buttons for common score ranges
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="200-249", callback_data="score_225"),
                InlineKeyboardButton(text="250-299", callback_data="score_275"),
            ],
            [
                InlineKeyboardButton(text="300-349", callback_data="score_325"),
                InlineKeyboardButton(text="350-399", callback_data="score_375"),
            ],
            [
                InlineKeyboardButton(text="400-449", callback_data="score_425"),
                InlineKeyboardButton(text="450+", callback_data="score_475"),
            ],
            [
                InlineKeyboardButton(
                    text="✏️ Ввести свой балл", callback_data="score_custom"
                ),
            ],
        ]
    )
    reply_text = (
        "*📊 Проверка потенциального места в рейтинге*\n\n"
        "Выберите диапазон баллов или введите свой балл:"
        if lang == "ru"
        else (
            "*📊 检查潜在排名*\n\n" "选择分数范围或输入您的分数:"
            if lang == "zh"
            else "*📊 Check Potential Ranking Position*\n\n"
            "Select a score range or enter your own score:"
        )
    )
    await message.reply(
        reply_text,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


@dp.message(
    lambda message: message.text and message.text.strip().lower() == "/updateprofile"
)
async def handle_update_profile(message: Message):
    """
    Handle the /updateprofile command to update user profile.

    This function initiates the profile update flow when a user clicks the
    "Update Profile" button. It first asks for language preference with buttons,
    then proceeds to collect updated profile information.

    Args:
        message (Message): The Telegram message containing the /updateprofile command

    Returns:
        None
    """
    uid = message.from_user.id

    # Initialize user profile if it doesn't exist
    if uid not in user_profiles:
        # Try to get from database first
        db_profile = await get_user_profile(uid)
        if db_profile:
            user_profiles[uid] = {
                "name": db_profile["name"],
                "interests": db_profile["interests"],
                "courses": db_profile["courses"],
            }
            # If user has a language preference in database, use it
            if db_profile.get("language"):
                user_languages[uid] = db_profile["language"]
        else:
            user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Create language selection buttons
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇨🇳 Chinese", callback_data="lang_zh"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="🇷🇺 Russian", callback_data="lang_ru"),
            ],
        ]
    )

    # Set user state to waiting for language selection with updating flag
    user_states[uid] = {"state": "waiting_for_language", "data": {"updating": True}}

    await message.reply(
        "Let's update your profile. First, please select your preferred language:\n\n"
        "让我们更新您的个人资料。首先，请选择您的首选语言：\n\n"
        "Давайте обновим ваш профиль. Сначала выберите предпочитаемый язык:",
        reply_markup=keyboard,
    )


# ─── Telegram: Callback Query Handlers ────────────────────────────────────────


@dp.callback_query(lambda c: c.data == "refresh_scores")
async def process_refresh_scores(callback_query: CallbackQuery):
    """
    Handle the 'Refresh Scores' button click event in the Telegram bot interface.

    This function is triggered when a user clicks the "Refresh Scores" button to get
    the most up-to-date exam scores or rankings. It provides real-time feedback by
    showing a temporary notification while fetching the latest data.

    The function performs the following steps:
    1. Shows a temporary notification to indicate the refresh operation is in progress
    2. Fetches the latest scores data from the source system using fetch_scores()
    3. Recreates the refresh button to allow subsequent refreshes
    4. Handles possible error conditions from the data fetch
    5. In the success case, formats the scores into a readable message
    6. Updates the original message with either the formatted scores or an error message

    Parameters:
        callback_query (CallbackQuery): The Telegram callback query object containing:
            - message: The original message with the button that was clicked
            - data: "refresh_scores" (filtered by the decorator)

    Side Effects:
        - Shows a temporary notification to the user
        - Makes an external API call to fetch the latest scores
        - Modifies the original message to display updated scores
        - Adds a refresh button to allow further refreshes

    Error Handling:
        - If the fetch_scores() function returns an error, displays the error message
          with an option to try again later
        - If no scores are available, displays a message indicating this

    Related Functions:
        - fetch_scores: Retrieves the latest scores data from the external system
        - handle_view_scores: Initial command handler that first displays the scores

    Returns:
        None
    """
    await callback_query.answer("Refreshing scores...")

    # Fetch latest scores
    scores = await fetch_scores()

    # Create refresh keyboard again
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Refresh Scores", callback_data="refresh_scores"
                )
            ]
        ]
    )

    if "error" in scores:
        await callback_query.message.edit_text(
            f"❌ *Error:* {scores['error']}\n\nPlease try again later.",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
    else:
        # Format scores for display
        scores_text = "*📊 Current Scores:*\n\n"
        for item in scores.get("items", []):
            scores_text += f"• {item['name']}: {item['score']}\n"

        if not scores.get("items", []):
            scores_text += "_No scores available at this time._\n"

        await callback_query.message.edit_text(
            scores_text, parse_mode="Markdown", reply_markup=keyboard
        )


@dp.callback_query(lambda c: c.data.startswith("subscribe_"))
async def process_subscription_response(callback_query: CallbackQuery):
    """
    Handle subscription preference selections from users in the Telegram bot interface.

    This function processes user responses to subscription prompts, allowing them to opt-in
    or opt-out of receiving notifications about exam results. It's triggered when a user
    clicks either the "Yes" or "No" button in the subscription dialog.

    The function performs the following steps:
    1. Extracts the user's choice from the callback data ("yes" or any other value)
    2. For "yes" responses:
    - [TODO] Stores the user's subscription preference in the database
    - Confirms successful subscription with a message
    3. For "no" responses:
    - Informs the user their subscription was cancelled
    - Reminds them they can subscribe later using the /subscribe command
    4. Acknowledges the callback query to remove the loading state from the clicked button

    Parameters:
        callback_query (CallbackQuery): The Telegram callback query object containing:
            - data: String starting with "subscribe_" followed by the user's choice
            - message: The original message with subscription options
            - from_user: Information about the user who clicked the button

    Side Effects:
        - Modifies the original message to show confirmation of user's subscription choice
        - Will save subscription preference in database once implemented
        - Acknowledges the callback query to remove loading state from the button

    Related Functions:
        - handle_subscribe: Initial command handler that presents the subscription options
        - Other notification-related functions that use subscription preferences

    Returns:
        None
    """
    choice = callback_query.data.split("_")[1]
    lang = user_languages.get(callback_query.from_user.id, "en")

    if choice == "yes":
        # TODO: Save subscription preference in database
        await callback_query.message.edit_text(
            (
                "✅ *Successfully subscribed to exam results!*\n\n"
                "You will receive notifications when new results are available."
                if lang == "en"
                else (
                    "✅ *成功订阅考试结果！*\n\n" "您将在有新结果时收到通知。"
                    if lang == "zh"
                    else "✅ *Успешно подписан на результаты экзаменов!*\n\n"
                    "Вы будете получать уведомления, когда будут доступны новые результаты."
                )
            ),
            parse_mode="Markdown",
        )
    else:
        await callback_query.message.edit_text(
            (
                "🚫 *Subscription cancelled.*\n\n"
                "You can subscribe anytime using the /subscribe command."
                if lang == "en"
                else (
                    "🚫 *订阅已取消。*\n\n" "您可以随时使用 /subscribe 命令订阅。"
                    if lang == "zh"
                    else "🚫 *Подписка отменена.*\n\n"
                    "Вы можете подписаться в любое время, используя команду /subscribe."
                )
            ),
            parse_mode="Markdown",
        )

    await callback_query.answer()


@dp.callback_query(lambda c: c.data.startswith("score_"))
async def process_score_callback(callback_query: CallbackQuery):
    """
    Handle score range selection for ranking calculation.

    This function processes a user's score range selection when they use the
    /checkscore command. It either uses a predefined score from the button
    or transitions to a state where the user can input a custom score.
    The function then calculates and displays where the user would rank
    in the current standings with that score.

    Args:
        callback_query (CallbackQuery): The callback query containing the selected score range

    Returns:
        None
    """
    uid = callback_query.from_user.id
    choice = callback_query.data.split("_")[1]

    logger.info(f"[SCORE] User {uid} selected score option: {choice}")

    if choice == "custom":
        # Set state to wait for custom score
        user_states[uid] = {"state": "waiting_for_score"}
        await callback_query.message.edit_text(
            "*📊 Введите свой балл*\n\n"
            "Пожалуйста, введите ваш балл за экзамен (число от 0 до 400):",
            parse_mode="Markdown",
        )
        await callback_query.answer()
        return

    # Process predefined score
    user_score = int(choice)
    await callback_query.answer(f"Проверяем место с баллом {user_score}...")

    # Fetch scores
    scores = await fetch_scores()

    if "error" in scores:
        logger.error(f"Error fetching scores: {scores['error']}")
        await callback_query.message.edit_text(
            f"❌ *Ошибка:* {scores['error']}\n\nПожалуйста, попробуйте позже.",
            parse_mode="Markdown",
        )
        return

    # Find where user would rank
    items = scores.get("items", [])

    # Calculate position
    position = 1
    for item in items:
        if user_score < item["score"]:
            position += 1
        else:
            break

    total_participants = len(items)

    try:
        await callback_query.message.edit_text(
            f"*📊 Ваше потенциальное место в рейтинге*\n\n"
            f"С баллом {user_score} вы бы заняли *{position} место* из {total_participants + 1} участников.",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception(f"Failed to send ranking response to user {uid}")
        # Try a simpler fallback
        await callback_query.message.edit_text(
            f"С баллом {user_score}: {position} из {total_participants+1}"
        )


@dp.callback_query(lambda c: c.data == "create_profile")
async def process_create_profile(callback_query: CallbackQuery):
    """
    Handle the 'Create Profile' button click event in the Telegram bot interface.

    This function is triggered when a user clicks the "Create Profile" button, initiating the
    profile creation workflow. It's the entry point for new users to set up their profile
    in the system.

    The function performs the following steps:
    1. Initializes a new empty user profile structure in the global user_profiles dictionary
    2. Presents a language selection interface with options for Chinese, English, and Russian
    3. Updates the user's state to indicate they're in the language selection phase
    4. Displays a trilingual welcome message with language selection buttons

    The selected language will determine the language used for subsequent interactions
    during the profile creation process.

    Parameters:
        callback_query (CallbackQuery): The Telegram callback query object containing:
            - from_user.id: The unique Telegram user ID
            - message: The original message with the button that was clicked
            - data: "create_profile" (filtered by the decorator)

    Side Effects:
        - Creates/resets an entry in user_profiles dictionary for the user
        - Updates user_states dictionary to track the user's profile creation progress
        - Modifies the original message to show language selection options
        - Acknowledges the callback query to remove loading state from the button

    Follow-up States:
        After this function, the user state is set to "waiting_for_language",
        which will be handled by the language selection callback handler.

    Related Functions:
        - process_language_selection: Handles the language selection callback
        - handle_name_input: Processes user's name after language selection
        - handle_interests_input: Processes user's interests input
        - handle_courses_input: Processes user's courses input
        - save_user_profile: Saves the completed profile to the database

    Returns:
        None
    """
    uid = callback_query.from_user.id

    # Initialize user profile
    user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Create language selection buttons
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇨🇳 Chinese", callback_data="lang_zh"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="🇷🇺 Russian", callback_data="lang_ru"),
            ],
        ]
    )

    # Set user state to waiting for language selection
    user_states[uid] = {"state": "waiting_for_language", "data": {}}

    await callback_query.message.edit_text(
        "👋 Welcome! Please select your preferred language:\n\n"
        "欢迎！请选择您的首选语言：\n\n"
        "Добро пожаловать! Пожалуйста, выберите предпочитаемый язык:",
        reply_markup=keyboard,
    )

    await callback_query.answer()


@dp.callback_query(lambda c: c.data == "update_profile")
async def process_update_profile(callback_query: CallbackQuery):
    """
    Handle the update_profile button click to update an existing profile.

    This function initiates the profile update flow when a user clicks the
    "Update Profile" button. It first asks for language preference with buttons,
    then proceeds to collect updated profile information.

    Args:
        callback_query (CallbackQuery): The callback query from the button click

    Returns:
        None
    """
    uid = callback_query.from_user.id

    # Initialize user profile if it doesn't exist
    if uid not in user_profiles:
        # Try to get from database first
        db_profile = await get_user_profile(uid)
        if db_profile:
            user_profiles[uid] = {
                "name": db_profile["name"],
                "interests": db_profile["interests"],
                "courses": db_profile["courses"],
            }
            # If user has a language preference in database, use it
            if db_profile.get("language"):
                user_languages[uid] = db_profile["language"]
        else:
            user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Create language selection buttons
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇨🇳 Chinese", callback_data="lang_zh"),
                InlineKeyboardButton(text="🇬🇧 English", callback_data="lang_en"),
            ],
            [
                InlineKeyboardButton(text="🇷🇺 Russian", callback_data="lang_ru"),
            ],
        ]
    )

    # Set user state to waiting for language selection with updating flag
    user_states[uid] = {"state": "waiting_for_language", "data": {"updating": True}}

    await callback_query.message.edit_text(
        "Let's update your profile. First, please select your preferred language:\n\n"
        "让我们更新您的个人资料。首先，请选择您的首选语言：\n\n"
        "Давайте обновим ваш профиль. Сначала выберите предпочитаемый язык:",
        reply_markup=keyboard,
    )

    await callback_query.answer()


# ─── Telegram: Incoming User Messages ────────────────────────────────────────


async def send_split_message(message: Message, text: str, parse_mode: str = None):
    """
    Split long messages and send them in parts to avoid Telegram's message size limitation.

    This function handles long messages that exceed Telegram's character limit by
    intelligently splitting them into multiple parts. It attempts to preserve paragraph
    structure when possible and falls back to sentence or word splitting when needed.

    Args:
        message (Message): The original Telegram message to reply to
        text (str): The text content to send, potentially exceeding Telegram's limits
        parse_mode (str, optional): The parse mode to use (e.g., 'Markdown', 'HTML')

    Returns:
        Message: The first message object that was sent as a reply

    The function adds continuation indicators to help users understand message order
    when splitting is required.
    """
    # Telegram's message size limit is around 4096 characters
    MAX_MESSAGE_LENGTH = 4000  # Using a slightly lower limit to be safe

    print(text)
    if len(text) <= MAX_MESSAGE_LENGTH:
        # If message is short enough, send it directly
        return await message.reply(text, parse_mode=parse_mode)

    # Message needs splitting
    parts = []
    current_part = ""

    # Try to split by paragraphs first
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit
        if len(current_part) + len(paragraph) + 2 > MAX_MESSAGE_LENGTH:
            # If current_part is already non-empty, add it to parts
            if current_part:
                parts.append(current_part)
                current_part = ""

            # If the paragraph itself is too long, split it by sentences
            if len(paragraph) > MAX_MESSAGE_LENGTH:
                sentences = paragraph.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if len(current_part) + len(sentence) + 1 > MAX_MESSAGE_LENGTH:
                        if current_part:
                            parts.append(current_part)
                            current_part = sentence
                        else:
                            # Even a single sentence is too long, split by words
                            words = sentence.split(" ")
                            for word in words:
                                if (
                                    len(current_part) + len(word) + 1
                                    > MAX_MESSAGE_LENGTH
                                ):
                                    parts.append(current_part)
                                    current_part = word
                                else:
                                    if current_part:
                                        current_part += " " + word
                                    else:
                                        current_part = word
                    else:
                        if current_part:
                            current_part += " " + sentence
                        else:
                            current_part = sentence
            else:
                current_part = paragraph
        else:
            # Add paragraph with appropriate separator
            if current_part:
                current_part += "\n\n" + paragraph
            else:
                current_part = paragraph

    # Add the last part if not empty
    if current_part:
        parts.append(current_part)

    # Send all parts
    first_message = None
    for i, part in enumerate(parts):
        if i == 0:
            first_message = await message.reply(part, parse_mode=parse_mode)
        else:
            await message.reply(
                f"(continued {i+1}/{len(parts)})\n\n{part}", parse_mode=parse_mode
            )

        # Add a small delay between messages to maintain order
        if i < len(parts) - 1:
            await asyncio.sleep(0.5)

    return first_message


async def clear_thread_if_needed(user_id, thread):
    """
    Periodically clear thread history if it gets too long to avoid context confusion.

    This function checks if a thread has accumulated too many messages and
    creates a fresh thread if needed, preserving only the user profile context.
    This helps prevent the model from getting confused by lengthy conversation
    history or repeating previous answers.

    Args:
        user_id (int): The user ID associated with the thread
        thread: The current thread object

    Returns:
        The thread object (either the original or a new one if cleared)
    """
    try:
        # Try to count messages in thread
        message_count = 0
        try:
            messages = thread.list_messages()
            message_count = len(messages)
        except Exception:
            pass

        # If thread has more than 20 messages, start a new one
        # This number can be adjusted based on performance
        if message_count > 20:
            logger.info(
                f"[THREAD] Clearing long thread for user {user_id} with {message_count} messages"
            )

            # Create a new thread
            new_thread = sdk.threads.create(ttl_days=7, expiration_policy="static")

            # Add user context to the new thread
            context_message = await build_user_context_message(user_id)
            if context_message:
                try:
                    new_thread.write(context_message)
                    logger.info(
                        f"[CONTEXT] Added profile context to new thread after clearing for user {user_id}"
                    )
                except Exception:
                    logger.warning(
                        f"[CONTEXT] Could not add context to new thread after clearing for user {user_id}"
                    )

            # Update the user's thread reference
            user_threads[user_id] = new_thread
            return new_thread

        # Return original thread if no clearing needed
        return thread
    except Exception as e:
        logger.exception(f"[ERROR] Failed to clear thread for user {user_id}")
        return thread  # Fall back to original thread


@dp.message()
async def handle_user_message(message: Message):
    """
    Main message handler for all incoming user messages.

    This function serves as the central dispatcher for processing user messages.
    It handles various states and conditions:
    1. Profile creation/state management
    2. Admin ticket management (claimed/pending)
    3. Explicit escalation commands
    4. LLM-based response generation
    5. Automatic escalation when needed

    The function implements a priority-based decision tree to determine the
    appropriate action for each incoming message.

    Args:
        message (Message): The Telegram message object from the user

    Returns:
        None
    """
    uid = message.from_user.id
    lang = user_languages.get(uid, "en")

    # Check for user profiling states
    if uid in user_states:
        state = user_states[uid].get("state")

        if state == "waiting_for_language":
            # If user somehow bypasses buttons and sends text, show buttons again
            keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🇨🇳 Chinese", callback_data="lang_zh"
                        ),
                        InlineKeyboardButton(
                            text="🇬🇧 English", callback_data="lang_en"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text="🇷🇺 Russian", callback_data="lang_ru"
                        ),
                    ],
                ]
            )
            await message.reply(
                (
                    "Please select your language using the buttons below:"
                    if lang == "en"
                    else (
                        "请使用下面的按钮选择您的语言:"
                        if lang == "zh"
                        else "Выберите язык, используя кнопки ниже:"
                    )
                ),
                reply_markup=keyboard,
            )
            return
        elif state == "waiting_for_name":
            await handle_name_input(message)
            return
        elif state == "waiting_for_interests":
            await handle_interests_input(message)
            return
        elif state == "waiting_for_courses":
            await handle_courses_input(message)
            return
        elif state == "waiting_for_score":
            await handle_manual_score_input(message)
            return

    # Skip other state-based conversations
    if uid in user_states:
        logger.info(f"[STATE] User {uid} in state {user_states[uid].get('state')}")
        return

    text = message.text or ""

    # Initialize dialog history for this user if it doesn't exist
    if uid not in dialog_history:
        dialog_history[uid] = []

    # Add user message to dialog history
    dialog_history[uid].append({"role": "user", "content": text})

    # 1) Already claimed → forward every message
    if uid in tickets_to_user.values():
        # find ticket_id by value
        ticket_id = next(k for k, v in tickets_to_user.items() if v == uid)
        await forward_to_admin(ticket_id, text)
        logger.info(f"[ADMIN] Forwarded message from user {uid} to ticket {ticket_id}")
        return await message.reply(
            (
                "📨 *Your message has been sent to the admin.*\n\nPlease wait for their response."
                if lang == "en"
                else (
                    "📨 *您的消息已发送给管理员。*\n\n请等待他们的回复。"
                    if lang == "zh"
                    else "📨 *Ваше сообщение было отправлено администратору.*\n\nПожалуйста, подождите, пока они ответят."
                )
            ),
            parse_mode="Markdown",
        )

    # 2) Escalated but not claimed → politely wait
    if uid in pending_tickets:
        logger.info(f"[TICKET] User {uid} has pending ticket {pending_tickets[uid]}")
        return await message.reply(
            (
                f"⏳ *Your ticket* `{pending_tickets[uid]}` *is waiting for an admin to claim.*\n\nThank you for your patience."
                if lang == "en"
                else (
                    f"⏳ *Ваш тикет* `{pending_tickets[uid]}` *ожидает, пока администратор его примет.*\n\nСпасибо за ваше терпение."
                    if lang == "ru"
                    else f"⏳ *您的工单* `{pending_tickets[uid]}` *正在等待管理员认领。*\n\n感谢您的耐心等待。"
                )
            ),
            parse_mode="Markdown",
        )

    # 3) Explicit escalation command
    if text.strip().lower() == "/escalate":
        try:
            ticket_id = await escalate_to_admin(uid, dialog_history[uid])
            logger.info(
                f"[ESCALATE] User {uid} manually escalated to ticket {ticket_id}"
            )
            return await message.reply(
                (
                    f"✅ *Escalated to admin.*\n\nTicket ID: `{ticket_id}`"
                    if lang == "en"
                    else (
                        f"✅ *Вы перешли к администратору.*\n\nID тикета: `{ticket_id}`"
                        if lang == "ru"
                        else f"✅ *已升级到管理员。*\n\n工单ID: `{ticket_id}`"
                    )
                ),
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception(f"[ERROR] Escalation failed for user {uid}")
            return await message.reply(
                (
                    "❌ *Unable to escalate your request at this time.*\n\nPlease try again later."
                    if lang == "en"
                    else (
                        "❌ *Не удалось эскалировать ваш запрос в данный момент.*\n\nПожалуйста, попробуйте позже."
                        if lang == "ru"
                        else "❌ *无法在此时升级您的请求。*\n\n请稍后再试。"
                    )
                ),
                parse_mode="Markdown",
            )

    # Check if this is a new conversation and the message starts with a greeting
    # This can help reset context more aggressively when a new conversation starts
    greeting_terms = [
        "привет",
        "здравствуй",
        "добрый день",
        "доброе утро",
        "добрый вечер",
        "hello",
        "hi",
        "你好",
        "您好",
        "こんにちは",
        "こんばんは",
        "こんばんは",
        "こんばんは",
        "こんばんは",
        "こんばんは",
    ]
    is_greeting = any(text.lower().startswith(term) for term in greeting_terms)

    # If this is a greeting and we have an existing thread, consider clearing it
    if is_greeting and uid in user_threads:
        logger.info(f"[THREAD] Detected greeting from user {uid}, may clear thread")
        # Force a thread reset more aggressively for greetings
        thread = user_threads.get(uid)
        if thread:
            try:
                # Create new thread regardless of message count
                new_thread = sdk.threads.create(ttl_days=7, expiration_policy="static")

                # Add user context to the new thread
                context_message = await build_user_context_message(user_id=uid)
                if context_message:
                    try:
                        new_thread.write(context_message)
                        logger.info(
                            f"[CONTEXT] Added profile context to new thread after greeting for user {uid}"
                        )
                    except Exception:
                        pass

                # Update the user's thread reference
                user_threads[uid] = new_thread
            except Exception:
                logger.exception(
                    f"[ERROR] Failed to reset thread after greeting for user {uid}"
                )

    # 4) LLM response
    max_llm_retries = 2
    for attempt in range(max_llm_retries):
        try:
            # Check if we need to clear the thread before proceeding
            if uid in user_threads:
                user_threads[uid] = await clear_thread_if_needed(uid, user_threads[uid])

            # Use Yandex Pro LLM assistant
            reply, flag = await call_yc(uid, text)

            # Successfully got a response, break the retry loop
            break
        except Exception as e:
            logger.exception(
                f"[ERROR] LLM call failed for user {uid} on attempt {attempt+1}/{max_llm_retries}"
            )

            # If this is the last attempt, use a fallback response
            if attempt == max_llm_retries - 1:
                # Default generic error message and trigger escalation
                reply = (
                    (
                        "I'm experiencing technical difficulties right now. You may want to try again later or use the /escalate command to reach a human admin."
                        if lang == "en"
                        else (
                            "我现在遇到了技术问题。您可能想稍后再试，或者使用 */escalate* 命令联系人工管理员。"
                            if lang == "zh"
                            else "У меня сейчас технические проблемы. Вы можете попробовать позже или использовать команду */escalate* для связи с администратором."
                        )
                    ),
                )
                flag = True
            else:
                # Wait briefly before retrying
                await asyncio.sleep(1)
                continue

    # 5) Model-triggered escalation hint
    if flag:
        try:
            ticket_id = await escalate_to_admin(uid, dialog_history[uid])
            logger.info(f"[ESCALATE] Auto-escalated user {uid} to ticket {ticket_id}")
            return await message.reply(
                (
                    f"✅ *I've escalated your request to a human admin.*\n\nTicket ID: `{ticket_id}`"
                    if lang == "en"
                    else (
                        (
                            f"✅ *Я передал ваш запрос администратору.*\n\nID тикета: `{ticket_id}`"
                            if lang == "ru"
                            else f"✅ *我已经将您的请求升级到人工管理员。*\n\n工单ID: `{ticket_id}`"
                        ),
                    )
                ),
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception(f"[ERROR] Auto-escalation failed for user {uid}")
            return await message.reply(
                (
                    "❌ *I tried to escalate your request, but there was an issue.*\n\nPlease try using the */escalate* command directly."
                    if lang == "en"
                    else (
                        (
                            "❌ *Я попытался эскалировать ваш запрос, но возникла проблема.*\n\nПожалуйста, попробуйте использовать команду */escalate* напрямую."
                            if lang == "ru"
                            else "❌ *我尝试升级您的请求，但遇到了问题。*\n\n请直接使用 */escalate* 命令。"
                        ),
                    )
                ),
            )

    # 6) Normal reply
    dialog_history[uid].append({"role": "assistant", "content": reply})

    # Use the split message function to handle potentially long replies
    await send_split_message(message, reply, parse_mode="Markdown")


# ─── Telegram: User Profiling Handlers ────────────────────────────────────


async def handle_name_input(message: Message):
    """
    Handle user's name input during the profile creation process.

    This function processes the user's name when they are in the name input state
    of the profile creation flow. It saves the name to the user's profile and
    transitions them to the next state (interests input).

    Args:
        message (Message): The Telegram message containing the user's name

    Returns:
        None
    """
    uid = message.from_user.id
    name = message.text.strip()

    # Save name to user profile
    user_profiles[uid]["name"] = name

    # Update state to ask for interests
    user_states[uid]["state"] = "waiting_for_interests"

    # Get user's language preference
    language = user_languages.get(uid, "en")

    updating = user_states[uid].get("data", {}).get("updating", False)

    # Prepare message based on language
    if language == "zh":
        if updating:
            interests_message = "您在考试或教育方面的主要兴趣是什么？"
        else:
            interests_message = (
                "您在考试或教育方面的主要兴趣是什么？（例如：数学，计算机科学，医学）"
            )
    elif language == "ru":
        if updating:
            interests_message = (
                "Каковы ваши основные интересы в экзаменах или образовании?"
            )
        else:
            interests_message = "Каковы ваши основные интересы в экзаменах или образовании? (например, математика, информатика, медицина)"
    else:  # Default to English
        if updating:
            interests_message = "What are your main interests in exams or education?"
        else:
            interests_message = "What are your main interests in exams or education? (e.g., math, computer science, medicine)"

    await message.reply(
        interests_message,
        parse_mode="Markdown",
    )


async def handle_interests_input(message: Message):
    """
    Handle user's interests input during the profile creation process.

    This function processes the user's academic/educational interests when they
    are in the interests input state of the profile creation flow. It saves the
    interests to the user's profile and transitions them to the next state
    (courses input).

    Args:
        message (Message): The Telegram message containing the user's interests

    Returns:
        None
    """
    uid = message.from_user.id
    interests = message.text.strip()

    # Save interests to user profile
    user_profiles[uid]["interests"] = interests

    # Update state to ask for courses
    user_states[uid]["state"] = "waiting_for_courses"

    # Get user's language preference
    language = user_languages.get(uid, "en")

    # Prepare message based on language
    if language == "zh":
        courses_message = "您目前正在学习或计划学习哪些课程或科目？"
    elif language == "ru":
        courses_message = (
            "Какие курсы или предметы вы сейчас изучаете или планируете изучать?"
        )
    else:  # Default to English
        courses_message = (
            "What courses or subjects are you currently studying or planning to take?"
        )

    await message.reply(
        courses_message,
        parse_mode="Markdown",
    )


async def handle_courses_input(message: Message):
    """
    Handle user's courses input and complete the profile creation process.

    This function processes the user's courses/subjects input, which is the final
    step in the profile creation flow. It saves the complete profile to the database,
    adds the profile to the conversation context, and initiates asynchronous
    summarization of the profile.

    Args:
        message (Message): The Telegram message containing the user's courses/subjects

    Returns:
        None
    """
    uid = message.from_user.id
    courses = message.text.strip()

    # Save courses to user profile
    user_profiles[uid]["courses"] = courses

    # Clear user state
    user_states.pop(uid, None)

    # Get complete user profile
    profile = user_profiles.get(uid, {})

    # Get language preference, default to English if not set
    language = user_languages.get(uid, "en")

    try:
        # Create profile text for history and summarization
        profile_text = f"User Profile:\nName: {profile.get('name', '')}\nInterests: {profile.get('interests', '')}\nCourses: {profile.get('courses', '')}\nLanguage: {language}"

        # Save profile to database (no agent response) with language
        save_success = await save_user_profile(
            uid,
            profile.get("name", ""),
            profile.get("interests", ""),
            profile.get("courses", ""),
            language=language,
        )

        if not save_success:
            logger.warning(f"[PROFILE] Failed to save profile for user {uid}")

        # Initialize dialog history for future conversations
        if uid not in dialog_history:
            dialog_history[uid] = []

        # Add the profile information to dialog history for context
        dialog_history[uid].append({"role": "system", "content": profile_text})
        logger.info(f"[PROFILE] Completed profile creation for user {uid}")

        # Start async summarization without waiting for it to complete
        asyncio.create_task(summarize_profile_async(uid, profile_text))

        # Prepare completion message based on selected language
        if language == "zh":
            completion_message = "谢谢您提供的信息！我能为您做什么？"
        elif language == "ru":
            completion_message = "Спасибо за информацию! Чем я могу вам помочь сегодня?"
        else:  # Default to English
            completion_message = (
                "Thanks for the information! How can I assist you today?"
            )

        # Just inform user that profiling is complete
        await message.reply(
            completion_message,
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.exception(f"[ERROR] Failed to process profile for user {uid}")
        await message.reply(
            "Thanks for sharing your information! How can I assist you today?",
            parse_mode="Markdown",
        )


# ─── Telegram: Manual Score Input Handler ────────────────────────────────────


async def handle_manual_score_input(message: Message):
    """
    Handle manually entered exam score after selecting the custom score option.

    This function processes a user-entered score value, validates that it's
    a number within the acceptable range (0-400), and then calculates and displays
    where the user would rank in the current standings with that score.

    Args:
        message (Message): The Telegram message containing the user's score

    Returns:
        None
    """
    uid = message.from_user.id
    text = message.text.strip() if message.text else ""
    lang = user_languages.get(uid, "en")

    logger.info(f"[SCORE] Processing manual score input from user {uid}: {text}")

    # Validate input is a number
    try:
        user_score = int(text)
        if user_score < 0 or user_score > 400:
            await message.reply(
                (
                    "❌ *Неверный балл*\n\n"
                    "Пожалуйста, введите допустимый балл за экзамен (число от 0 до 400):"
                    if lang == "ru"
                    else (
                        "❌ *Invalid score*\n\nPlease enter a valid exam score (number between 0 and 400):"
                        if lang == "en"
                        else "❌ *無效分數*\n\n請輸入有效的考試分數（0到400之間的數字）："
                    )
                ),
                parse_mode="Markdown",
            )
            return
    except ValueError:
        await message.reply(
            (
                "❌ *Неверный ввод*\n\n" "Пожалуйста, введите число от 0 до 400:"
                if lang == "ru"
                else (
                    "❌ *Invalid input*\n\nPlease enter a number between 0 and 400:"
                    if lang == "en"
                    else "❌ *無效輸入*\n\n請輸入0到400之間的數字："
                )
            ),
            parse_mode="Markdown",
        )
        return

    # Clear user state
    user_states.pop(uid, None)

    # Fetch current scores
    logger.info(
        f"[SCORE] Fetching scores for user {uid} with manual score {user_score}"
    )
    scores = await fetch_scores()

    if "error" in scores:
        logger.error(f"Error fetching scores: {scores['error']}")
        await message.reply(
            (
                (
                    f"❌ *Ошибка:* {scores['error']}\n\nПожалуйста, попробуйте позже."
                    if lang == "ru"
                    else (
                        f"❌ *Error:* {scores['error']}\n\nPlease try again later."
                        if lang == "en"
                        else f"❌ *錯誤:* {scores['error']}\n\n請稍後再試。"
                    )
                ),
            ),
            parse_mode="Markdown",
        )
        return

    # Find where user would rank
    items = scores.get("items", [])
    logger.info(f"Found {len(items)} items in score data")

    # Insert user's score into the sorted list and determine position
    position = 1
    for item in items:
        if user_score < item["score"]:
            position += 1
        else:
            break

    total_participants = len(items)
    logger.info(
        f"User {uid} with score {user_score} would rank {position} out of {total_participants+1}"
    )

    # Simple response with just the position
    try:
        await message.reply(
            (
                f"*📊 Ваше потенциальное место в рейтинге*\n\n"
                f"С баллом {user_score} вы бы заняли *{position} место* из {total_participants + 1} участников."
                if lang == "ru"
                else (
                    (
                        f"*📊 Your potential ranking*\n\n"
                        f"With a score of {user_score}, you would rank *{position} out of {total_participants + 1} participants."
                    )
                    if lang == "en"
                    else (
                        f"*📊 您的潛在排名*\n\n"
                        f"以 {user_score} 分，您将排名 *{position} 位* 从 {total_participants + 1} 名参与者中。"
                    )
                )
            ),
            parse_mode="Markdown",
        )
        logger.info(f"Successfully sent ranking response to user {uid}")
    except Exception as e:
        logger.exception(f"Failed to send ranking response to user {uid}")
        # Try a simpler message as fallback
        await message.reply(
            (
                f"С баллом {user_score}: {position} из {total_participants+1}"
                if lang == "ru"
                else (
                    (
                        f"With a score of {user_score}: {position} out of {total_participants+1}"
                        if lang == "en"
                        else f"以 {user_score} 分：{position} 从 {total_participants+1} 名参与者中。"
                    ),
                )
            ),
        )


# Also add a profile command to view saved profile
@dp.message(lambda message: message.text and message.text.strip().lower() == "/profile")
async def handle_profile(message: Message):
    """
    Handle the /profile command to display a user's saved profile.

    This function retrieves and displays the user's profile information from
    the database, including their name, interests, courses, and last update time.
    For admins, it also shows the AI-generated profile summary if available.

    Args:
        message (Message): The Telegram message containing the /profile command

    Returns:
        None
    """
    uid = message.from_user.id
    is_admin = uid in ADMIN_USER_IDS
    lang = user_languages.get(uid, "en")

    # Try to get from database first
    db_profile = await get_user_profile(uid)

    if db_profile:
        # Format profile information
        profile_text_en = (
            f"*👤 Your Profile*\n\n"
            f"*Name:* {db_profile['name']}\n"
            f"*Interests:* {db_profile['interests']}\n"
            f"*Courses:* {db_profile['courses']}\n\n"
            f"*Last updated:* {db_profile['updated_at']}"
        )
        profile_text_ru = (
            f"*👤 Ваш профиль*\n\n"
            f"*Имя:* {db_profile['name']}\n"
            f"*Интересы:* {db_profile['interests']}\n"
            f"*Курсы:* {db_profile['courses']}\n\n"
            f"*Последнее обновление:* {db_profile['updated_at']}"
        )
        profile_text_zh = (
            f"*👤 您的個人資料*\n\n"
            f"*姓名:* {db_profile['name']}\n"
            f"*興趣:* {db_profile['interests']}\n"
            f"*課程:* {db_profile['courses']}\n\n"
            f"*最後更新:* {db_profile['updated_at']}"
        )
        profile_text = (
            profile_text_en
            if lang == "en"
            else profile_text_ru if lang == "ru" else profile_text_zh
        )

        # Show summary to admins if available
        if is_admin and db_profile.get("profile_summary"):
            profile_text += f"\n\n*AI Summary:* {db_profile['profile_summary']}"

        # Add update profile button
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✏️ Update Profile", callback_data="update_profile"
                    )
                ]
            ]
        )

        await message.reply(profile_text, parse_mode="Markdown", reply_markup=keyboard)
    else:
        # Check in-memory profiles as fallback
        if uid in user_profiles:
            profile = user_profiles[uid]
            profile_text_en = (
                f"*👤 Your Profile (Not Saved)*\n\n"
                f"*Name:* {profile.get('name', 'Not provided')}\n"
                f"*Interests:* {profile.get('interests', 'Not provided')}\n"
                f"*Courses:* {profile.get('courses', 'Not provided')}"
            )
            profile_text_ru = (
                f"*👤 Ваш профиль (Не сохранен)*\n\n"
                f"*Имя:* {profile.get('name', 'Не предоставлено')}\n"
                f"*Интересы:* {profile.get('interests', 'Не предоставлено')}\n"
                f"*Курсы:* {profile.get('courses', 'Не предоставлено')}"
            )
            profile_text_zh = (
                f"*👤 您的個人資料 (未保存)*\n\n"
                f"*姓名:* {profile.get('name', '未提供')}\n"
                f"*興趣:* {profile.get('interests', '未提供')}\n"
                f"*課程:* {profile.get('courses', '未提供')}"
            )
            profile_text = (
                profile_text_en
                if lang == "en"
                else profile_text_ru if lang == "ru" else profile_text_zh
            )

            await message.reply(profile_text, parse_mode="Markdown")
        else:
            # No profile found
            await message.reply(
                (
                    "❓ *No profile found*\n\n"
                    "You haven't created a profile yet. Use the /start command to create one."
                    if lang == "en"
                    else (
                        "❓ *找不到個人資料*\n\n"
                        "您還沒有創建個人資料。使用 /start 命令創建一個。"
                        if lang == "zh"
                        else "❓ Вы еще не создали профиль. Используйте команду /start для создания."
                    )
                ),
                parse_mode="Markdown",
            )


@dp.message(
    lambda message: message.text and message.text.strip().lower() == "/updateprofile"
)
async def handle_update_profile(message: Message):
    """
    Handle the /updateprofile command to update user profile.

    This function initiates the profile update flow when a user clicks the
    "Update Profile" button. It sets the appropriate state to begin collecting
    updated profile information, starting with the name.

    Args:
        message (Message): The Telegram message containing the /updateprofile command

    Returns:
        None
    """
    uid = message.from_user.id
    lang = user_languages.get(uid, "en")

    # Initialize user profile if it doesn't exist
    if uid not in user_profiles:
        # Try to get from database first
        db_profile = await get_user_profile(uid)
        if db_profile:
            user_profiles[uid] = {
                "name": db_profile["name"],
                "interests": db_profile["interests"],
                "courses": db_profile["courses"],
            }
        else:
            user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Set user state to begin updating profile
    user_states[uid] = {"state": "waiting_for_name", "data": {"updating": True}}

    await message.reply(
        (
            "Let's update your information. What is your name?"
            if lang == "en"
            else (
                "Давайте обновим вашу информацию. Как вас зовут?"
                if lang == "ru"
                else "讓我們更新您的信息。您的名字是什麼？"
            )
        ),
        parse_mode="Markdown",
    )


@dp.callback_query(lambda c: c.data == "update_profile")
async def process_update_profile(callback_query: CallbackQuery):
    """
    Handle the update_profile button click to update an existing profile.

    This function initiates the profile update flow when a user clicks the
    "Update Profile" button. It sets the appropriate state to begin collecting
    updated profile information, starting with the name.

    Args:
        callback_query (CallbackQuery): The callback query from the button click

    Returns:
        None
    """
    uid = callback_query.from_user.id
    lang = user_languages.get(uid, "en")

    # Initialize user profile if it doesn't exist
    if uid not in user_profiles:
        # Try to get from database first
        db_profile = await get_user_profile(uid)
        if db_profile:
            user_profiles[uid] = {
                "name": db_profile["name"],
                "interests": db_profile["interests"],
                "courses": db_profile["courses"],
            }
        else:
            user_profiles[uid] = {"name": "", "interests": "", "courses": ""}

    # Set user state to begin updating profile
    user_states[uid] = {"state": "waiting_for_name", "data": {"updating": True}}

    await callback_query.message.edit_text(
        (
            "Let's update your information. What is your name?"
            if lang == "en"
            else (
                "Давайте обновим вашу информацию. Как вас зовут?"
                if lang == "ru"
                else "讓我們更新您的信息。您的名字是什麼？"
            )
        ),
        parse_mode="Markdown",
    )

    await callback_query.answer()


# ─── Flask: Health Check ───────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for monitoring system status.

    This endpoint provides a simple health check that returns the number of
    active threads and users to verify the service is operational. It's
    designed for use with monitoring systems and load balancers to confirm
    the bot is functioning normally.

    Returns:
        Flask response: JSON containing connection count and HTTP 200 status code
    """
    return jsonify({"connections": len(user_threads)}), 200


@app.route("/claim", methods=["POST"])
def receive_claim():
    """
    Endpoint for Admin-Bot to claim a user's ticket.

    This endpoint allows the Admin-Bot to claim a ticket for a specific user,
    establishing a direct communication channel between the admin and user.
    Once claimed, the user's messages will be forwarded to the admin until
    the ticket is closed.

    Expected JSON payload:
        {
            "ticket_id": "ticket identifier",
            "user_id": optional user ID if not retrievable from pending tickets
        }

    Returns:
        tuple: (JSON response, HTTP status code)
    """
    data = request.json or {}
    ticket_id = data.get("ticket_id")
    user_id = data.get("user_id") or pending_ticket_to_user.get(ticket_id)
    if not ticket_id or user_id is None:
        return jsonify(error="Missing ticket_id or user_id"), 400

    logger.info("Incoming /claim for ticket %s: user_id %s", ticket_id, user_id)
    # clear any pending state
    pending_tickets.pop(user_id, None)
    pending_ticket_to_user.pop(ticket_id, None)
    # mark claimed
    tickets_to_user[ticket_id] = user_id
    return jsonify(status="claimed"), 200


@app.route("/reply", methods=["POST"])
def receive_admin_reply():
    """
    Endpoint for Admin-Bot to send replies to users or close tickets.

    This endpoint receives admin replies to claimed tickets and forwards them
    to the corresponding user. It also handles ticket closure notifications,
    which resets the ticket state and returns the user to normal LLM interaction.

    Expected JSON payload:
        {
            "ticket_id": "ticket identifier",
            "text": "admin's reply message"
        }

    Returns:
        tuple: (JSON response, HTTP status code)

    If the message text begins with "🔒 Ticket" or contains "closed by", the
    function treats it as a ticket closure notification and cleans up ticket state.
    """
    data = request.json or {}
    ticket_id = data.get("ticket_id")
    text = data.get("text")
    if not ticket_id or text is None:
        return jsonify(error="Missing ticket_id or text"), 400

    user_id = tickets_to_user.get(ticket_id)
    if not user_id:
        return jsonify(error="Unknown or unclaimed ticket"), 404

    # --- detect ticket-close notification ---
    if text.startswith("🔒 Ticket") or "closed by" in text.lower():
        # Clear out claimed-state so future user messages go to LLM again
        tickets_to_user.pop(ticket_id, None)
        pending_tickets.pop(user_id, None)
        pending_ticket_to_user.pop(ticket_id, None)

        # Notify the user we're back to normal
        asyncio.run_coroutine_threadsafe(
            bot.send_message(
                chat_id=user_id,
                text="✅ *Your ticket has been closed.* I'm back to my normal assistant mode!",
                parse_mode="Markdown",
            ),
            polling_loop,
        )
        return jsonify(status="closed"), 200

    # --- otherwise, normal admin reply ---
    try:
        # Format the admin reply
        admin_reply = f"👤 *Admin Reply:*\n\n{text}"

        # Check if the message is too long
        MAX_MESSAGE_LENGTH = 4000
        if len(admin_reply) <= MAX_MESSAGE_LENGTH:
            # If short enough, send as a single message
            asyncio.run_coroutine_threadsafe(
                bot.send_message(
                    chat_id=user_id,
                    text=admin_reply,
                    parse_mode="Markdown",
                ),
                polling_loop,
            )
        else:
            # If too long, split the reply and send multiple messages
            # First send header
            asyncio.run_coroutine_threadsafe(
                bot.send_message(
                    chat_id=user_id,
                    text="👤 *Admin Reply:*",
                    parse_mode="Markdown",
                ),
                polling_loop,
            )

            # Then split and send the content
            parts = []
            current_part = ""

            # Try to split by paragraphs first
            paragraphs = text.split("\n\n")

            for paragraph in paragraphs:
                if len(current_part) + len(paragraph) + 2 > MAX_MESSAGE_LENGTH:
                    if current_part:
                        parts.append(current_part)
                        current_part = paragraph
                    else:
                        parts.append(paragraph)
                else:
                    if current_part:
                        current_part += "\n\n" + paragraph
                    else:
                        current_part = paragraph

            # Add the last part if not empty
            if current_part:
                parts.append(current_part)

            # Send all parts
            for i, part in enumerate(parts):
                part_text = part
                if len(parts) > 1:
                    part_text = f"(part {i+1}/{len(parts)})\n\n{part}"

                asyncio.run_coroutine_threadsafe(
                    bot.send_message(
                        chat_id=user_id,
                        text=part_text,
                        parse_mode="Markdown",
                    ),
                    polling_loop,
                )

                # Small delay between messages to maintain order
                if i < len(parts) - 1:
                    asyncio.run_coroutine_threadsafe(
                        asyncio.sleep(0.5),
                        polling_loop,
                    )

    except Exception:
        logger.exception(
            "Failed to send message to user %s for ticket %s", user_id, ticket_id
        )
        return jsonify(error="Send failed"), 500

    return jsonify(status="sent"), 200


# ─── Admin Commands ────────────────────────────────────────────────────────

# List of admin user IDs
ADMIN_USER_IDS = (
    list(map(int, os.getenv("ADMIN_USER_IDS", "").split(",")))
    if os.getenv("ADMIN_USER_IDS")
    else []
)


@dp.message(
    lambda message: message.text
    and message.text.strip().lower() == "/admin"
    and message.from_user.id in ADMIN_USER_IDS
)
async def handle_admin(message: Message):
    """
    Handle the /admin command to show admin-only options and commands.

    This function displays a list of administrative commands available only
    to users whose IDs are in the ADMIN_USER_IDS list. These commands provide
    access to bot statistics, user profile summaries, and other admin functions.

    Args:
        message (Message): The Telegram message containing the /admin command

    Returns:
        None

    The function verifies that the user is in the admin list before responding,
    as enforced by the message filter in the decorator.
    """
    admin_text = (
        "*🔐 Admin Commands*\n\n"
        "• /viewsummaries - View all user profile summaries\n"
        "• /getstats - View bot usage statistics\n"
    )

    await message.reply(admin_text, parse_mode="Markdown")


@dp.message(
    lambda message: message.text
    and message.text.strip().lower() == "/viewsummaries"
    and message.from_user.id in ADMIN_USER_IDS
)
async def handle_view_summaries(message: Message):
    """
    Handle the /viewsummaries admin command to list all profile summaries.

    This function retrieves and displays AI-generated summaries of all user
    profiles from the database. It's restricted to admin users only and
    provides a comprehensive view of user profiles for administrative purposes.

    Args:
        message (Message): The Telegram message containing the /viewsummaries command

    Returns:
        None

    The function handles long responses by splitting the output into multiple
    messages if needed. It retrieves the 20 most recently updated profiles
    with summaries.
    """
    try:
        async with get_db_connection() as db:
            async with db.execute(
                """
            SELECT user_id, name, profile_summary, updated_at
            FROM user_profiles 
            WHERE profile_summary IS NOT NULL AND profile_summary != ""
            ORDER BY updated_at DESC
            LIMIT 20
            """
            ) as cursor:
                profiles = await cursor.fetchall()

        if not profiles:
            await message.reply(
                "No profile summaries available yet.", parse_mode="Markdown"
            )
            return

        summaries_text = "*📊 User Profile Summaries*\n\n"

        for profile in profiles:
            user_id, name, summary, updated_at = (
                profile["user_id"],
                profile["name"],
                profile["profile_summary"],
                profile["updated_at"],
            )
            summaries_text += f"*👤 User:* {name} (ID: {user_id})\n"
            summaries_text += f"*Updated:* {updated_at}\n"
            summaries_text += f"*Summary:* {summary}\n\n"

            # Split message if it gets too long
            if (
                summaries_text
                and summaries_text != "*📊 User Profile Summaries (continued)*\n\n"
                and len(summaries_text) > 3500
            ):
                await message.reply(summaries_text, parse_mode="Markdown")
                summaries_text = "*📊 User Profile Summaries (continued)*\n\n"

        # Send remaining text
        if (
            summaries_text
            and summaries_text != "*📊 User Profile Summaries (continued)*\n\n"
        ):
            await message.reply(summaries_text, parse_mode="Markdown")

    except Exception as e:
        logger.exception("Failed to retrieve profile summaries")
        await message.reply(
            "❌ Error retrieving profile summaries.", parse_mode="Markdown"
        )


@dp.message(
    lambda message: message.text
    and message.text.strip().lower() == "/getstats"
    and message.from_user.id in ADMIN_USER_IDS
)
async def handle_get_stats(message: Message):
    """
    Handle the /getstats admin command to show bot usage statistics.

    This function gathers and displays comprehensive statistics about the bot's
    usage and performance. It retrieves database stats, memory usage, and active
    user metrics. This command is restricted to admin users only.

    Args:
        message (Message): The Telegram message containing the /getstats command

    Returns:
        None

    The statistics include:
    - Database metrics: total profiles, profiles with summaries, newest profile, latest update
    - Memory metrics: active threads, active dialogs, pending escalations, claimed tickets
    """
    try:
        # Get database stats
        async with get_db_connection() as db:
            # Count total profiles
            async with db.execute(
                "SELECT COUNT(*) as count FROM user_profiles"
            ) as cursor:
                result = await cursor.fetchone()
                total_profiles = result["count"] if result else 0

            # Count profiles with summaries
            async with db.execute(
                "SELECT COUNT(*) as count FROM user_profiles WHERE profile_summary IS NOT NULL AND profile_summary != ''"
            ) as cursor:
                result = await cursor.fetchone()
                profiles_with_summaries = result["count"] if result else 0

            # Get newest profile
            async with db.execute(
                "SELECT created_at FROM user_profiles ORDER BY created_at DESC LIMIT 1"
            ) as cursor:
                newest_profile_result = await cursor.fetchone()
                newest_profile = (
                    newest_profile_result["created_at"]
                    if newest_profile_result
                    else "N/A"
                )

            # Get most recently updated profile
            async with db.execute(
                "SELECT updated_at FROM user_profiles ORDER BY updated_at DESC LIMIT 1"
            ) as cursor:
                latest_update_result = await cursor.fetchone()
                latest_update = (
                    latest_update_result["updated_at"]
                    if latest_update_result
                    else "N/A"
                )

        # Memory stats
        active_threads = len(user_threads)
        active_dialogs = len(dialog_history)
        pending_escalations = len(pending_tickets)
        claimed_tickets = len(tickets_to_user)

        stats_text = (
            "*📊 Bot Statistics*\n\n"
            f"*Database:*\n"
            f"• Total profiles: {total_profiles}\n"
            f"• Profiles with summaries: {profiles_with_summaries}\n"
            f"• Newest profile: {newest_profile}\n"
            f"• Latest update: {latest_update}\n\n"
            f"*Memory:*\n"
            f"• Active threads: {active_threads}\n"
            f"• Active dialogs: {active_dialogs}\n"
            f"• Pending escalations: {pending_escalations}\n"
            f"• Claimed tickets: {claimed_tickets}\n"
        )

        await message.reply(stats_text, parse_mode="Markdown")
    except Exception as e:
        logger.exception("Failed to retrieve bot statistics")
        await message.reply("❌ Error retrieving statistics.", parse_mode="Markdown")


# ─── Bootstrap: Run Flask + Telegram Polling ─────────────────────────────────

if __name__ == "__main__":
    # 1) Create & install asyncio loop
    polling_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(polling_loop)

    # Initialize database
    polling_loop.run_until_complete(setup_database())

    # Initialize Yandex Pro LLM assistant if not in debug mode
    if not DEBUG_MODE:
        try:
            ind_ru = get_or_create_index(INDEX_NAME_RU)
            yandex_assistant_ru = get_or_create_assistant(ind_ru, ASSISTANT_NAME_RU)
            yandex_summarizer = get_or_create_assistant(0, ASSISTANT_NAME_SUM)
            ind_en = get_or_create_index(INDEX_NAME_EN)
            yandex_assistant_en = get_or_create_assistant(ind_en, ASSISTANT_NAME_EN)
            ind_zh = get_or_create_index(INDEX_NAME_ZH)
            yandex_assistant_zh = get_or_create_assistant(ind_zh, ASSISTANT_NAME_ZH)

            if yandex_assistant_ru:
                logger.info(
                    f"Successfully initialized Yandex Pro LLM assistant: {yandex_assistant_ru.id}"
                )
            else:
                logger.error(
                    "Could not initialize Yandex Pro LLM assistant - returned None"
                )
                logger.warning("Falling back to DEBUG_MODE")
                DEBUG_MODE = True

            if yandex_assistant_en:
                logger.info(
                    f"Successfully initialized Yandex Pro LLM assistant: {yandex_assistant_en.id}"
                )
            else:
                logger.error(
                    "Could not initialize Yandex Pro LLM assistant - returned None"
                )
                logger.warning("Falling back to DEBUG_MODE")
                DEBUG_MODE = True

            if yandex_assistant_zh:
                logger.info(
                    f"Successfully initialized Yandex Pro LLM assistant: {yandex_assistant_zh.id}"
                )
            else:
                logger.error(
                    "Could not initialize Yandex Pro LLM assistant - returned None"
                )
                logger.warning("Falling back to DEBUG_MODE")
                DEBUG_MODE = True

            if yandex_summarizer:
                logger.info(
                    f"Successfully initialized summarizer assistant: {yandex_summarizer.id}"
                )
            else:
                logger.warning(
                    "Could not initialize summarizer assistant - summaries will be disabled"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Yandex Pro LLM assistant: {str(e)}")
            logger.warning("Falling back to DEBUG_MODE")
            DEBUG_MODE = True

    # 2) Start Flask in background thread
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, use_reloader=False),
        daemon=True,
    ).start()
    logger.info("Flask listening on port 8080")

    # 3) Signal handlers for graceful shutdown
    def _shutdown(_sig, _frame):
        """
        Handle graceful shutdown of the bot when receiving termination signals.

        This function is registered as a signal handler for SIGINT and SIGTERM.
        It performs clean shutdown operations including:
        1. Deleting all Yandex thread objects to free resources
        2. Closing all database connections properly
        3. Stopping the main event loop

        Args:
            _sig: Signal number (unused but required by signal handler interface)
            _frame: Current stack frame (unused but required by signal handler interface)

        Returns:
            None
        """
        logger.info("Shutdown signal received; stopping.")

        # Clean up Yandex threads
        if not DEBUG_MODE:
            for uid, thread in user_threads.items():
                try:
                    thread.delete()
                    logger.info(f"Deleted thread for user {uid}")
                except Exception:
                    logger.exception(f"Failed to delete thread for user {uid}")

        # Close database connections
        polling_loop.run_until_complete(db_manager.close_all())

        polling_loop.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 4) Start Aiogram polling
    polling_loop.create_task(dp.start_polling(bot, skip_updates=True))
    polling_loop.run_forever()
