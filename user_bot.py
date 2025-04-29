#!/usr/bin/env python3
import os
import asyncio
import logging
import threading
import signal

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from yandex_cloud_ml_sdk import YCloudML
import aiohttp

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


# ─── Telegram: Incoming User Messages ────────────────────────────────────────


@dp.message()
async def handle_user_message(message: Message):
    uid = message.from_user.id
    text = message.text or ""
    hist = dialog_history.setdefault(uid, [])
    hist.append({"role": "user", "content": text})

    # 1) Already claimed → forward every message
    if uid in tickets_to_user.values():
        # find ticket_id by value
        ticket_id = next(k for k, v in tickets_to_user.items() if v == uid)
        await forward_to_admin(ticket_id, text)
        return await message.reply("📨 Sent to admin.")

    # 2) Escalated but not claimed → politely wait
    if uid in pending_tickets:
        return await message.reply(
            f"⏳ Your ticket `{pending_tickets[uid]}` is waiting for an admin to claim. Please hold on."
        )

    # 3) Explicit escalation command
    if text.strip().lower() == "/escalate":
        try:
            ticket_id = await escalate_to_admin(uid, hist)
            return await message.reply(f"✅ Escalated to admin (ticket `{ticket_id}`).")
        except Exception:
            logger.exception("Escalation failed for user %s", uid)
            return await message.reply("❌ Failed to escalate. Please try again later.")

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
            return await message.reply("❌ I'm having trouble right now.")

        # 6) Model-triggered escalation hint
        if "[escalate]" in reply.lower():
            try:
                ticket_id = await escalate_to_admin(uid, hist)
                return await message.reply(
                    f"✅ Model requested escalation (ticket `{ticket_id}`)."
                )
            except Exception:
                logger.exception("Escalation failed for user %s in model hint", uid)
                return await message.reply(
                    "❌ Failed to escalate. Please try again later."
                )

    # 7) Normal reply
    hist.append({"role": "assistant", "content": reply})
    await message.reply(reply)


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

    logger.info("Incoming /reply for ticket %s: %r", ticket_id, text)
    try:
        asyncio.run_coroutine_threadsafe(
            bot.send_message(chat_id=user_id, text=f"👤 Admin: {text}"), polling_loop
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

    # 3) Start Aiogram polling
    polling_loop.create_task(dp.start_polling(bot, skip_updates=True))
    logger.info("Telegram polling started")

    # 4) Graceful shutdown
    def _shutdown(_sig, _frame):
        logger.info("Shutdown signal received; stopping.")
        polling_loop.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 5) Run loop forever
    polling_loop.run_forever()
