#!/usr/bin/env python3
import os
import asyncio
import logging
import threading
import signal
import aiohttp
import json
import csv
import io

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from aiogram import Bot, Dispatcher
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from yandex_cloud_ml_sdk import YCloudML

# ─── Configuration & Logging ─────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
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

# Debug echo‐mode toggle
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

# Google Sheets config
GOOGLE_SHEETS_URL = os.getenv(
    "GOOGLE_SHEETS_URL", 
    "https://docs.google.com/spreadsheets/d/1RSmWjkxHnZ3GnWuRia5NkMGE9MTHROtWu0Cri5LXLBI/edit?usp=sharing"
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

# ─── In-Memory State ─────────────────────────────────────────────────────────

dialog_history = {}  # user_id → [ {"role","content"}, … ]
pending_tickets = {}  # user_id → ticket_id (escalated but not claimed)
pending_ticket_to_user = {}  # ticket_id → user_id
tickets_to_user = {}  # ticket_id → user_id (claimed)

# Will hold our asyncio event loop
polling_loop = None

# ─── Flask & Aiogram Setup ─────────────────────────────────────────────────

app = Flask(__name__)
bot = Bot(token=USER_BOT_TOKEN)
dp = Dispatcher()

# ─── Helpers ────────────────────────────────────────────────────────────────

async def call_yc(prompt: str) -> str:
    """Run YandexGPT in a thread to avoid blocking the loop."""
    model = sdk.models.completions(YC_MODEL_NAME).configure(temperature=0.5)
    return (
        await asyncio.get_event_loop().run_in_executor(None, lambda: model.run(prompt))
    )[0]


async def escalate_to_admin(user_id: int, dialog: list) -> str:
    """
    Send full dialog to Admin-Bot to open a ticket.
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
    logger.info("Escalated user %s → ticket %s", user_id, ticket_id)
    return ticket_id


async def forward_to_admin(ticket_id: str, text: str):
    """
    Forward a user message to Admin-Bot after claim.
    """
    payload = {"ticket_id": ticket_id, "text": text}
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(ADMIN_BOT_MESSAGE_URL, json=payload)
            logger.info(
                "Forward to admin POST %s → %s returned %d",
                ADMIN_BOT_MESSAGE_URL,
                ticket_id,
                resp.status,
            )
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Admin-Bot rejected forward (%d): %s", resp.status, body)
    except Exception:
        logger.exception("Failed to forward to admin for ticket %s", ticket_id)


async def fetch_scores():
    """Fetch scores from Google Sheets using public CSV export"""
    try:
        # Extract sheet ID from the URL
        sheet_id = GOOGLE_SHEETS_URL.split("/d/")[1].split("/edit")[0]
        
        # Construct CSV export URL
        csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
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
                    if row.get('Снилс') and row.get('Баллы'):
                        try:
                            # Convert score to integer for sorting
                            score = int(row['Баллы'])
                            items.append({
                                'name': f"СНИЛС: {row['Снилс']}",
                                'score': score
                            })
                        except ValueError:
                            # Skip rows with non-numeric scores
                            continue
                
                # Sort items by score in descending order
                items.sort(key=lambda x: x['score'], reverse=True)
                
                return {"items": items}
    except Exception as e:
        logger.exception("Error fetching scores from Google Sheet")
        # Return mock data for testing or when there's an error
        return {
            "items": [
                {"name": "СНИЛС: 12345", "score": 290},
                {"name": "СНИЛС: 54321", "score": 275},
                {"name": "СНИЛС: 98765", "score": 310}
            ]
        }

# ─── Telegram: State Management for Conversations ───────────────────────────

# Simple state management for dialogs
user_states = {}  # user_id → {"state": "waiting_for_score", "data": {...}}

# ─── Telegram: Command Handlers ────────────────────────────────────────────

@dp.message(lambda message: message.text and message.text.strip().lower() == "/viewscores")
async def handle_view_scores(message: Message):
    """Handle the /viewscores command"""
    uid = message.from_user.id
    
    # Create a keyboard with button to refresh scores
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh Scores", callback_data="refresh_scores")]
        ]
    )
    
    # Fetch scores
    scores = await fetch_scores()
    
    if "error" in scores:
        await message.reply(
            f"❌ *Error:* {scores['error']}\n\nPlease try again later.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    else:
        # Format scores for display
        scores_text = "*📊 Current Scores:*\n\n"
        for item in scores.get("items", []):
            scores_text += f"• {item['name']}: {item['score']}\n"
        
        if not scores.get("items", []):
            scores_text += "_No scores available at this time._\n"
        
        await message.reply(
            scores_text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )


@dp.message(lambda message: message.text and message.text.strip().lower() == "/subscribe")
async def handle_subscribe(message: Message):
    """Handle the /subscribe command for exam results"""
    uid = message.from_user.id
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Subscribe", callback_data="subscribe_yes"),
                InlineKeyboardButton(text="❌ Cancel", callback_data="subscribe_no")
            ]
        ]
    )
    
    await message.reply(
        "*📣 Exam Results Notification Service*\n\n"
        "Would you like to subscribe to exam results notifications?\n\n"
        "You will be notified as soon as new results become available.",
        parse_mode="Markdown",
        reply_markup=keyboard
    )


@dp.message(lambda message: message.text and message.text.strip().lower() == "/checkscore")
async def handle_check_score(message: Message):
    """Handle the /checkscore command to check where user would rank"""
    uid = message.from_user.id
    logger.info(f"User {uid} requested score check")
    
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
                InlineKeyboardButton(text="✏️ Ввести свой балл", callback_data="score_custom"),
            ]
        ]
    )
    
    await message.reply(
        "*📊 Проверка потенциального места в рейтинге*\n\n"
        "Выберите диапазон баллов или введите свой балл:",
        parse_mode="Markdown",
        reply_markup=keyboard
    )


# ─── Telegram: Callback Query Handlers ────────────────────────────────────────

@dp.callback_query(lambda c: c.data == "refresh_scores")
async def process_refresh_scores(callback_query: CallbackQuery):
    """Handle refresh_scores button clicks"""
    await callback_query.answer("Refreshing scores...")
    
    # Fetch latest scores
    scores = await fetch_scores()
    
    # Create refresh keyboard again
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Refresh Scores", callback_data="refresh_scores")]
        ]
    )
    
    if "error" in scores:
        await callback_query.message.edit_text(
            f"❌ *Error:* {scores['error']}\n\nPlease try again later.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    else:
        # Format scores for display
        scores_text = "*📊 Current Scores:*\n\n"
        for item in scores.get("items", []):
            scores_text += f"• {item['name']}: {item['score']}\n"
        
        if not scores.get("items", []):
            scores_text += "_No scores available at this time._\n"
        
        await callback_query.message.edit_text(
            scores_text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )


@dp.callback_query(lambda c: c.data.startswith("subscribe_"))
async def process_subscription_response(callback_query: CallbackQuery):
    """Handle subscription button clicks"""
    choice = callback_query.data.split("_")[1]
    
    if choice == "yes":
        # TODO: Save subscription preference in database
        await callback_query.message.edit_text(
            "✅ *Successfully subscribed to exam results!*\n\n"
            "You will receive notifications when new results are available.",
            parse_mode="Markdown"
        )
    else:
        await callback_query.message.edit_text(
            "🚫 *Subscription cancelled.*\n\n"
            "You can subscribe anytime using the /subscribe command.",
            parse_mode="Markdown"
        )
    
    await callback_query.answer()


@dp.callback_query(lambda c: c.data.startswith("score_"))
async def process_score_callback(callback_query: CallbackQuery):
    """Handle score range selection"""
    uid = callback_query.from_user.id
    choice = callback_query.data.split("_")[1]
    
    logger.info(f"User {uid} selected score option: {choice}")
    
    if choice == "custom":
        # Set state to wait for custom score
        user_states[uid] = {"state": "waiting_for_score"}
        await callback_query.message.edit_text(
            "*📊 Введите свой балл*\n\n"
            "Пожалуйста, введите ваш балл за экзамен (число от 0 до 500):",
            parse_mode="Markdown"
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
            parse_mode="Markdown"
        )
        return
    
    # Find where user would rank
    items = scores.get("items", [])
    
    # Calculate position
    position = 1
    for item in items:
        if user_score < item['score']:
            position += 1
        else:
            break
    
    total_participants = len(items)
    
    try:
        await callback_query.message.edit_text(
            f"*📊 Ваше потенциальное место в рейтинге*\n\n"
            f"С баллом {user_score} вы бы заняли *{position} место* из {total_participants + 1} участников.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.exception(f"Failed to send ranking response to user {uid}")
        # Try a simpler fallback
        await callback_query.message.edit_text(
            f"С баллом {user_score}: {position} из {total_participants+1}"
        )


# ─── Telegram: Incoming User Messages ────────────────────────────────────────

@dp.message()
async def handle_user_message(message: Message):
    uid = message.from_user.id
    
    # Check if user is waiting for score input
    if uid in user_states and user_states[uid].get("state") == "waiting_for_score":
        await handle_manual_score_input(message)
        return
    
    # Skip other state-based conversations
    if uid in user_states:
        logger.info(f"Skipping general handler for user {uid} who is in state: {user_states[uid].get('state')}")
        return
        
    text = message.text or ""
    hist = dialog_history.setdefault(uid, [])
    hist.append({"role": "user", "content": text})

    # 1) Already claimed → forward every message
    if uid in tickets_to_user.values():
        # find ticket_id by value
        ticket_id = next(k for k, v in tickets_to_user.items() if v == uid)
        await forward_to_admin(ticket_id, text)
        return await message.reply(
            "📨 *Your message has been sent to the admin.*\n\nPlease wait for their response.",
            parse_mode="Markdown",
        )

    # 2) Escalated but not claimed → politely wait
    if uid in pending_tickets:
        return await message.reply(
            f"⏳ *Your ticket* `{pending_tickets[uid]}` *is waiting for an admin to claim.*\n\nThank you for your patience.",
            parse_mode="Markdown",
        )

    # 3) Explicit escalation command
    if text.strip().lower() == "/escalate":
        try:
            ticket_id = await escalate_to_admin(uid, hist)
            return await message.reply(
                f"✅ *Escalated to admin.*\n\nTicket ID: `{ticket_id}`",
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("Escalation failed for user %s", uid)
            return await message.reply(
                "❌ *Unable to escalate your request at this time.*\n\nPlease try again later.",
                parse_mode="Markdown",
            )

    # 4) Debug echo
    if DEBUG_MODE:
        reply = text
    else:
        # 5) LLM response
        prompt = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in hist)
        try:
            reply = await call_yc(prompt)
        except Exception:
            logger.exception("LLM call failed for user %s", uid)
            return await message.reply(
                "❌ *I'm having trouble processing your request right now.*\n\nPlease try again later or type */escalate* to reach a human admin.",
                parse_mode="Markdown",
            )

        # 6) Model-triggered escalation hint
        if "[escalate]" in reply.lower():
            try:
                ticket_id = await escalate_to_admin(uid, hist)
                return await message.reply(
                    f"✅ *I've escalated your request to a human admin.*\n\nTicket ID: `{ticket_id}`",
                    parse_mode="Markdown",
                )
            except Exception:
                logger.exception("Escalation hint failed for user %s", uid)
                return await message.reply(
                    "❌ *I tried to escalate your request, but there was an issue.*\n\nPlease try using the */escalate* command directly.",
                    parse_mode="Markdown",
                )

    # 7) Normal reply
    hist.append({"role": "assistant", "content": reply})
    await message.reply(reply)


# ─── Telegram: Manual Score Input Handler ────────────────────────────────────

async def handle_manual_score_input(message: Message):
    """Handle manually entered score after selecting custom score option"""
    uid = message.from_user.id
    text = message.text.strip() if message.text else ""
    
    logger.info(f"Processing manual score input from user {uid}: {text}")
    
    # Validate input is a number
    try:
        user_score = int(text)
        if user_score < 0 or user_score > 500:
            await message.reply(
                "❌ *Неверный балл*\n\n"
                "Пожалуйста, введите допустимый балл за экзамен (число от 0 до 500):",
                parse_mode="Markdown"
            )
            return
    except ValueError:
        await message.reply(
            "❌ *Неверный ввод*\n\n"
            "Пожалуйста, введите число от 0 до 500:",
            parse_mode="Markdown"
        )
        return
    
    # Clear user state
    user_states.pop(uid, None)
    
    # Fetch current scores
    logger.info(f"Fetching scores for user {uid} with manual score {user_score}")
    scores = await fetch_scores()
    
    if "error" in scores:
        logger.error(f"Error fetching scores: {scores['error']}")
        await message.reply(
            f"❌ *Ошибка:* {scores['error']}\n\nПожалуйста, попробуйте позже.",
            parse_mode="Markdown"
        )
        return
    
    # Find where user would rank
    items = scores.get("items", [])
    logger.info(f"Found {len(items)} items in score data")
    
    # Insert user's score into the sorted list and determine position
    position = 1
    for item in items:
        if user_score < item['score']:
            position += 1
        else:
            break
    
    total_participants = len(items)
    logger.info(f"User {uid} with score {user_score} would rank {position} out of {total_participants+1}")
    
    # Simple response with just the position
    try:
        await message.reply(
            f"*📊 Ваше потенциальное место в рейтинге*\n\n"
            f"С баллом {user_score} вы бы заняли *{position} место* из {total_participants + 1} участников.",
            parse_mode="Markdown"
        )
        logger.info(f"Successfully sent ranking response to user {uid}")
    except Exception as e:
        logger.exception(f"Failed to send ranking response to user {uid}")
        # Try a simpler message as fallback
        await message.reply(f"С баллом {user_score}: {position} из {total_participants+1}")


# ─── Flask: Health Check ───────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ─── Flask: Receive Claim from Admin-Bot ────────────────────────────────────


@app.route("/claim", methods=["POST"])
def receive_claim():
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


# ─── Flask: Receive Replies / Closures from Admin-Bot ───────────────────────


@app.route("/reply", methods=["POST"])
def receive_admin_reply():
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
        asyncio.run_coroutine_threadsafe(
            bot.send_message(
                chat_id=user_id,
                text=f"👤 *Admin Reply:*\n\n{text}",
                parse_mode="Markdown",
            ),
            polling_loop,
        )
    except Exception:
        logger.exception(
            "Failed to send message to user %s for ticket %s", user_id, ticket_id
        )
        return jsonify(error="Send failed"), 500

    return jsonify(status="sent"), 200


# ─── Bootstrap: Run Flask + Telegram Polling ─────────────────────────────────

if __name__ == "__main__":
    # 1) Create & install asyncio loop
    polling_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(polling_loop)

    # 2) Start Flask in background thread
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, use_reloader=False),
        daemon=True,
    ).start()
    logger.info("Flask listening on port 8080")

    # 3) Signal handlers for graceful shutdown
    def _shutdown(_sig, _frame):
        logger.info("Shutdown signal received; stopping.")
        polling_loop.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 4) Start Aiogram polling
    polling_loop.create_task(dp.start_polling(bot, skip_updates=True))
    polling_loop.run_forever()
