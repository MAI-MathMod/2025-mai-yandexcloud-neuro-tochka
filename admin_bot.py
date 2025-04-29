#!/usr/bin/env python3
import os
import sys
import threading
import asyncio
import signal
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from flask import Flask, request, jsonify
import aiohttp

from aiogram import Bot, Dispatcher
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
    BotCommand,
    BotCommandScopeDefault,
)
from aiogram.enums import ParseMode
from aiogram.filters import Command

from jose import jwt, JWTError
from asyncio import run_coroutine_threadsafe

# ─── Configuration & Logging ─────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ADMIN_BOT_TOKEN = os.getenv("ADMIN_BOT_TOKEN")
# Ensure these point at your User-Bot service host/container
USER_BOT_REPLY_URL = os.getenv("USER_BOT_REPLY_URL", "http://user-bot:8080/reply")
USER_BOT_CLAIM_URL = os.getenv("USER_BOT_CLAIM_URL", "http://user-bot:8080/claim")
ADMIN_BOT_HEALTH_URL = os.getenv("ADMIN_BOT_HEALTH_URL", "http://localhost:8096/health")
USER_BOT_HEALTH_URL = os.getenv("USER_BOT_HEALTH_URL", "http://localhost:8080/health")
SUPERUSER_ID = int(os.getenv("SUPERUSER_ID", "0"))
JWT_SECRET = os.getenv("JWT_SECRET")
DB_PATH = os.getenv("DB_PATH", "admin_bot.db")
TOKEN_EXP_MINUTES = int(os.getenv("TOKEN_EXPIRATION_MINUTES", "30"))

if not (ADMIN_BOT_TOKEN and JWT_SECRET and SUPERUSER_ID):
    logger.error("Missing required environment variables; exiting.")
    sys.exit(1)

# ─── In-Memory State ─────────────────────────────────────────────────────────

tickets = {}  # ticket_id → { user_id, dialog, status, admin_id }
pending_notifications = []  # queue of notifications for superuser
active_conversations = {}  # admin_id → ticket_id
browsing_sessions = {}  # admin_id → { tickets: [...], index: int }

# ─── Flask & Aiogram Setup ──────────────────────────────────────────────────

app = Flask(__name__)
bot = Bot(token=ADMIN_BOT_TOKEN)
dp = Dispatcher()
looptask = None  # will hold our asyncio event loop

# ─── Database Helpers ───────────────────────────────────────────────────────


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        cur = db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (
              id INTEGER PRIMARY KEY,
              telegram_id INTEGER UNIQUE NOT NULL,
              is_superuser BOOLEAN NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tokens (
              token TEXT PRIMARY KEY,
              telegram_id INTEGER NOT NULL,
              salt TEXT NOT NULL,
              expires_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "INSERT OR IGNORE INTO admins (telegram_id, is_superuser) VALUES (?,1)",
            (SUPERUSER_ID,),
        )
        db.commit()


def is_admin(tg_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as db:
        return bool(
            db.execute(
                "SELECT 1 FROM admins WHERE telegram_id = ?", (tg_id,)
            ).fetchone()
        )


def is_superuser(tg_id: int) -> bool:
    return tg_id == SUPERUSER_ID


init_db()

# ─── Flask Endpoints ────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/tickets", methods=["POST"])
def receive_ticket():
    data = request.json or {}
    user_id = data.get("user_id")
    dialog = data.get("dialog")
    if not user_id or not isinstance(dialog, list):
        return jsonify(error="Missing or invalid user_id/dialog"), 400

    ticket_id = str(uuid.uuid4())
    tickets[ticket_id] = {
        "user_id": user_id,
        "dialog": dialog,
        "status": "open",
        "admin_id": None,
    }
    pending_notifications.append(
        {
            "chat_id": SUPERUSER_ID,
            "ticket_id": ticket_id,
            "text": f"🆕 New ticket `{ticket_id}` from user `{user_id}`.",
        }
    )
    logger.info("Ticket %s created by user %s", ticket_id, user_id)
    return jsonify(ticket_id=ticket_id), 200


@app.route("/message", methods=["POST"])
def receive_user_followup():
    data = request.json or {}
    ticket_id = data.get("ticket_id")
    text = data.get("text")
    if not ticket_id or text is None:
        return jsonify(error="Missing ticket_id or text"), 400

    info = tickets.get(ticket_id)
    if not info:
        return jsonify(error="Unknown ticket_id"), 404
    admin_id = info.get("admin_id")
    if not admin_id:
        return jsonify(error="Ticket not claimed"), 400

    # forward follow-up to claimed admin
    run_coroutine_threadsafe(
        bot.send_message(
            chat_id=admin_id, text=f"📨 Follow-up from user `{ticket_id}`:\n{text}"
        ),
        looptask,
    )
    return jsonify(status="forwarded"), 200


# ─── Aiogram Command Handlers ──────────────────────────────────────────────


@dp.message(Command(commands=["issue_token"]))
async def cmd_issue_token(message: Message):
    if not is_superuser(message.from_user.id):
        return await message.reply("❌ Only superuser can issue tokens.")
    salt = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=TOKEN_EXP_MINUTES)
    payload = {
        "telegram_id": message.from_user.id,
        "salt": salt,
        "iat": now.timestamp(),
        "exp": exp.timestamp(),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT INTO tokens (token, telegram_id, salt, expires_at) VALUES (?,?,?,?)",
            (token, message.from_user.id, salt, exp.isoformat()),
        )
        db.commit()

    await message.reply("✅ Admin token created:")
    await message.answer(token)


@dp.message(Command(commands=["auth"]))
async def cmd_auth(message: Message):
    parts = message.text.strip().split(maxsplit=1)
    if len(parts) != 2:
        return await message.reply("Usage: /auth <token>")
    token = parts[1]

    with sqlite3.connect(DB_PATH) as db:
        row = db.execute(
            "SELECT telegram_id, salt, expires_at FROM tokens WHERE token = ?", (token,)
        ).fetchone()

    if not row:
        return await message.reply("❌ Invalid or revoked token.")
    stored_tg, salt, exp_str = row
    if datetime.now(timezone.utc) > datetime.fromisoformat(exp_str):
        return await message.reply("❌ Token expired.")
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        return await message.reply("❌ Invalid token signature.")
    if decoded.get("telegram_id") != stored_tg or decoded.get("salt") != salt:
        return await message.reply("❌ Token mismatch.")

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT OR IGNORE INTO admins (telegram_id, is_superuser) VALUES (?,0)",
            (message.from_user.id,),
        )
        db.execute("DELETE FROM tokens WHERE token = ?", (token,))
        db.commit()

    await message.reply("✅ You are now authenticated as admin.")


@dp.message(Command(commands=["open"]))
async def cmd_open(message: Message):
    if not is_admin(message.from_user.id):
        return await message.reply("❌ Unauthorized.")
    lines = [
        f'🆔 *{tid}*  👤 `{info["user_id"]}`'
        for tid, info in tickets.items()
        if info["status"] == "open"
    ]
    await message.reply(
        "\n".join(lines) or "No open tickets.", parse_mode=ParseMode.MARKDOWN
    )


@dp.message(Command(commands=["pending"]))
async def cmd_pending(message: Message):
    if not is_admin(message.from_user.id):
        return await message.reply("❌ Unauthorized.")
    lines = [
        f'🆔 *{tid}*  👤 `{info["user_id"]}`'
        for tid, info in tickets.items()
        if info["status"] == "pending"
    ]
    await message.reply(
        "\n".join(lines) or "No pending tickets.", parse_mode=ParseMode.MARKDOWN
    )


@dp.message(Command(commands=["closed"]))
async def cmd_closed(message: Message):
    if not is_admin(message.from_user.id):
        return await message.reply("❌ Unauthorized.")
    lines = [
        f"🆔 *{tid}*" for tid, info in tickets.items() if info["status"] == "closed"
    ]
    await message.reply(
        "\n".join(lines) or "No closed tickets.", parse_mode=ParseMode.MARKDOWN
    )


@dp.message(Command(commands=["status"]))
async def cmd_status(message: Message):
    if not is_admin(message.from_user.id):
        return await message.reply("❌ Unauthorized.")
    async with aiohttp.ClientSession() as sess:
        try:
            r1 = await sess.get(ADMIN_BOT_HEALTH_URL)
            a_stat = "Online" if r1.status == 200 else f"Error {r1.status}"
        except Exception as e:
            a_stat = f"Down ({e})"
        try:
            r2 = await sess.get(USER_BOT_HEALTH_URL)
            u_stat = "Online" if r2.status == 200 else f"Error {r2.status}"
        except Exception as e:
            u_stat = f"Down ({e})"
    await message.reply(
        f"🤖 *Service Status*\n• Admin Bot: {a_stat}\n• User Bot: {u_stat}",
        parse_mode=ParseMode.MARKDOWN,
    )


@dp.message(Command(commands=["browse"]))
async def cmd_browse(message: Message):
    admin_id = message.from_user.id
    if not is_admin(admin_id):
        return await message.reply("❌ Unauthorized.")
    open_ids = [tid for tid, info in tickets.items() if info["status"] == "open"]
    if not open_ids:
        return await message.reply("No open tickets.")
    browsing_sessions[admin_id] = {"tickets": open_ids, "index": 0}
    await send_ticket_card(admin_id, open_ids[0], is_new=True)


async def send_ticket_card(admin_id: int, ticket_id: str, is_new=False, edit_msg=None):
    session = browsing_sessions[admin_id]
    idx = session["tickets"].index(ticket_id)
    info = tickets[ticket_id]
    conv = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in info["dialog"]
    )
    text = f"🆔 *{ticket_id}*  👤 `{info['user_id']}`\n\n*Conversation:*\n{conv}"

    nav_buttons = []
    if idx > 0:
        nav_buttons.append(
            InlineKeyboardButton(text="⬅️ Prev", callback_data="browse:prev")
        )
    if idx < len(session["tickets"]) - 1:
        nav_buttons.append(
            InlineKeyboardButton(text="Next ➡️", callback_data="browse:next")
        )
    claim_btn = InlineKeyboardButton(text="🗂️ Claim", callback_data=f"claim:{ticket_id}")
    kb = InlineKeyboardMarkup(inline_keyboard=[nav_buttons, [claim_btn]])

    if is_new:
        await bot.send_message(
            admin_id, text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb
        )
    else:
        await edit_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)


@dp.callback_query(lambda c: c.data and c.data.startswith("browse:"))
async def cb_browse(call: CallbackQuery):
    admin_id = call.from_user.id
    session = browsing_sessions.get(admin_id)
    if not session:
        return await call.answer("No browsing session.", show_alert=True)

    direction = call.data.split(":", 1)[1]
    idx = session["index"]
    if direction == "prev" and idx > 0:
        session["index"] -= 1
    elif direction == "next" and idx < len(session["tickets"]) - 1:
        session["index"] += 1
    else:
        return await call.answer()

    new_tid = session["tickets"][session["index"]]
    await send_ticket_card(admin_id, new_tid, is_new=False, edit_msg=call.message)
    await call.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("claim:"))
async def cb_claim_ticket(call: CallbackQuery):
    admin_id = call.from_user.id
    tid = call.data.split(":", 1)[1]
    info = tickets.get(tid)
    if not info or info["status"] != "open":
        return await call.answer("Cannot claim this ticket.", show_alert=True)

    tickets[tid]["status"] = "pending"
    tickets[tid]["admin_id"] = admin_id
    active_conversations[admin_id] = tid
    browsing_sessions.pop(admin_id, None)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        await sess.post(
            USER_BOT_CLAIM_URL, json={"ticket_id": tid, "user_id": info["user_id"]}
        )

    close_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🛑 Close Ticket", callback_data=f"close:{tid}")]
        ]
    )
    await call.message.edit_text(
        f"🎫 You claimed ticket *{tid}*. Send messages below or close when done.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=close_kb,
    )
    await call.answer("Ticket claimed.")


@dp.callback_query(lambda c: c.data and c.data.startswith("close:"))
async def cb_close(call: CallbackQuery):
    admin_id = call.from_user.id
    tid = call.data.split(":", 1)[1]
    if active_conversations.get(admin_id) != tid:
        return await call.answer("❌ Not on that ticket.", show_alert=True)

    tickets[tid]["status"] = "closed"
    active_conversations.pop(admin_id, None)

    await call.message.edit_reply_markup()
    await call.message.reply(
        f"✅ Ticket *{tid}* closed.", parse_mode=ParseMode.MARKDOWN
    )

    username = call.from_user.username or call.from_user.full_name
    msg = f"🔒 Ticket `{tid}` closed by @{username}."
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        try:
            resp = await sess.post(
                USER_BOT_REPLY_URL,
                json={"ticket_id": tid, "text": msg},
            )
            logger.info(
                "POST %s → ticket %s closed notification returned %d",
                USER_BOT_REPLY_URL,
                tid,
                resp.status,
            )
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Close notification returned %d: %s", resp.status, body)
            if resp.status == 404:
                await sess.post(
                    USER_BOT_CLAIM_URL,
                    json={"ticket_id": tid, "user_id": tickets[tid]["user_id"]},
                )
                retry_resp = await sess.post(
                    USER_BOT_REPLY_URL,
                    json={"ticket_id": tid, "text": msg},
                )
                logger.info(
                    "Retry close POST %s → ticket %s returned %d",
                    USER_BOT_REPLY_URL,
                    tid,
                    retry_resp.status,
                )
        except Exception:
            logger.exception("Failed to send close notification for ticket %s", tid)

    await call.answer()


@dp.message()
async def handle_admin_chat(message: Message):
    admin_id = message.from_user.id
    if not message.text or message.text.startswith("/"):
        return
    tid = active_conversations.get(admin_id)
    if not tid:
        return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        try:
            resp = await sess.post(
                USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": message.text}
            )
            logger.info(
                "POST %s → ticket %s returned %d", USER_BOT_REPLY_URL, tid, resp.status
            )
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Admin-Bot /reply returned %d: %s", resp.status, body)
            if resp.status == 404:
                await sess.post(
                    USER_BOT_CLAIM_URL,
                    json={"ticket_id": tid, "user_id": tickets[tid]["user_id"]},
                )
                retry_resp = await sess.post(
                    USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": message.text}
                )
                logger.info(
                    "Retry POST %s → ticket %s returned %d",
                    USER_BOT_REPLY_URL,
                    tid,
                    retry_resp.status,
                )
        except Exception:
            logger.exception("Failed to forward to User-Bot for ticket %s", tid)


async def main():
    commands = [
        BotCommand(command="issue_token", description="Generate an admin JWT token"),
        BotCommand(command="auth", description="Authenticate as admin"),
        BotCommand(command="open", description="List open tickets"),
        BotCommand(command="pending", description="List pending tickets"),
        BotCommand(command="closed", description="List closed tickets"),
        BotCommand(command="status", description="Check services health"),
        BotCommand(command="browse", description="Browse open tickets"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())
    await dp.start_polling(bot, skip_updates=True)


def shutdown(_sig, _frame):
    logger.info("Shutting down…")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8096, use_reloader=False),
        daemon=True,
    ).start()
    logger.info("Flask listening on port 8096")

    looptask = asyncio.new_event_loop()
    asyncio.set_event_loop(looptask)
    looptask.run_until_complete(main())
