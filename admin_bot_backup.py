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
pending_notifications = []  # queue for superuser ticket alerts
# Now track for each admin: list of claimed tickets + current active
active_conversations = (
    {}
)  # admin_id → { "claimed": [ticket_ids], "current": ticket_id or None }
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
            )"""
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tokens (
              token TEXT PRIMARY KEY,
              telegram_id INTEGER NOT NULL,
              salt TEXT NOT NULL,
              expires_at TEXT NOT NULL
            )"""
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

    # forward follow-up to claimed admin (on their current context UI)
    run_coroutine_threadsafe(
        bot.send_message(
            chat_id=admin_id, text=f"📨 Follow-up from user `{ticket_id}`:\n{text}"
        ),
        looptask,
    )
    return jsonify(status="forwarded"), 200


# ─── Helpers for Admin UI ───────────────────────────────────────────────────


async def send_open_ticket_card(
    admin_id: int, ticket_id: str, is_new=False, edit_msg=None
):
    """Original browsing card for open tickets."""
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


async def send_claimed_ticket_card(admin_id: int, ticket_id: str, edit_msg=None):
    """Show the conversation and controls for a claimed ticket."""
    info = tickets[ticket_id]
    conv = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in info["dialog"]
    )
    text = f"🎫 *Current Ticket:* `{ticket_id}`  👤 `{info['user_id']}`\n\n*Conversation so far:*\n{conv}"

    buttons = [
        [InlineKeyboardButton(text="🛑 Close", callback_data=f"close:{ticket_id}")],
        [
            InlineKeyboardButton(
                text="⏸️ Suspend", callback_data=f"suspend:{ticket_id}"
            ),
            InlineKeyboardButton(text="🔄 Switch", callback_data="switch:list"),
        ],
    ]
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)

    if edit_msg:
        await edit_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
    else:
        await bot.send_message(
            admin_id, text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb
        )


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
    await send_open_ticket_card(admin_id, open_ids[0], is_new=True)


# ─── Callback: Browsing Open Tickets ────────────────────────────────────────


@dp.callback_query(lambda c: c.data and c.data.startswith("browse:"))
async def cb_browse(call: CallbackQuery):
    admin_id = call.from_user.id
    if admin_id not in browsing_sessions:
        return await call.answer("No browsing session.", show_alert=True)
    session = browsing_sessions[admin_id]
    direction = call.data.split(":", 1)[1]
    idx = session["index"]
    if direction == "prev" and idx > 0:
        session["index"] -= 1
    elif direction == "next" and idx < len(session["tickets"]) - 1:
        session["index"] += 1
    else:
        return await call.answer()
    new_tid = session["tickets"][session["index"]]
    await send_open_ticket_card(admin_id, new_tid, is_new=False, edit_msg=call.message)
    await call.answer()


# ─── Callback: Claiming Tickets ─────────────────────────────────────────────


@dp.callback_query(lambda c: c.data and c.data.startswith("claim:"))
async def cb_claim_ticket(call: CallbackQuery):
    admin_id, tid = call.from_user.id, call.data.split(":", 1)[1]
    info = tickets.get(tid)
    if not info or info["status"] != "open":
        return await call.answer("Cannot claim this ticket.", show_alert=True)

    # mark pending & record admin
    tickets[tid]["status"] = "pending"
    tickets[tid]["admin_id"] = admin_id

    # add to this admin's claimed pool
    session = active_conversations.setdefault(
        admin_id, {"claimed": [], "current": None}
    )
    session["claimed"].append(tid)
    session["current"] = tid
    browsing_sessions.pop(admin_id, None)

    # notify user-bot of claim
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        await sess.post(
            USER_BOT_CLAIM_URL, json={"ticket_id": tid, "user_id": info["user_id"]}
        )

    # show the claimed-ticket card
    await send_claimed_ticket_card(admin_id, tid, edit_msg=call.message)
    await call.answer("Ticket claimed.")


# ─── Callback: Close Tickets ────────────────────────────────────────────────


@dp.callback_query(lambda c: c.data and c.data.startswith("close:"))
async def cb_close(call: CallbackQuery):
    admin_id, tid = call.from_user.id, call.data.split(":", 1)[1]
    session = active_conversations.get(admin_id)
    if not session or session["current"] != tid:
        return await call.answer("❌ Not on that ticket.", show_alert=True)

    # update status
    tickets[tid]["status"] = "closed"
    tickets[tid]["admin_id"] = None

    # remove from admin's pool
    session["claimed"].remove(tid)
    session["current"] = None

    # remove buttons on this card
    await call.message.edit_reply_markup()
    await call.message.reply(
        f"✅ Ticket *{tid}* closed.", parse_mode=ParseMode.MARKDOWN
    )

    # send closure to user-bot
    username = call.from_user.username or call.from_user.full_name
    msg = f"🔒 Ticket `{tid}` closed by @{username}."
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        try:
            resp = await sess.post(
                USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": msg}
            )
            logger.info("Close → user-bot returned %d", resp.status)
            # handle 404 by re-claim+retry
            if resp.status == 404:
                await sess.post(
                    USER_BOT_CLAIM_URL,
                    json={"ticket_id": tid, "user_id": tickets[tid]["user_id"]},
                )
                await sess.post(
                    USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": msg}
                )
        except Exception:
            logger.exception("Failed to send close notification for %s", tid)

    await call.answer()


# ─── Callback: Suspend Current Ticket ───────────────────────────────────────


@dp.callback_query(lambda c: c.data and c.data.startswith("suspend:"))
async def cb_suspend(call: CallbackQuery):
    admin_id, tid = call.from_user.id, call.data.split(":", 1)[1]
    session = active_conversations.get(admin_id)
    if not session or session["current"] != tid:
        return await call.answer("❌ Not on that ticket.", show_alert=True)
    # suspend: clear current selection
    session["current"] = None
    # remove buttons
    await call.message.edit_reply_markup()
    await call.message.reply(
        f"😴 *Ticket `{tid}` suspended.*\n\nUse 🔄 Switch to pick another ticket.",
        parse_mode=ParseMode.MARKDOWN,
    )
    await call.answer("Suspended.")


# ─── Callback: Switch Between Claimed Tickets ──────────────────────────────


@dp.callback_query(lambda c: c.data and c.data.startswith("switch:"))
async def cb_switch(call: CallbackQuery):
    admin_id = call.from_user.id
    session = active_conversations.get(admin_id)
    if not session:
        return await call.answer("You have no claimed tickets.", show_alert=True)

    _, key = call.data.split(":", 1)
    # list menu
    if key == "list":
        if not session["claimed"]:
            return await call.answer("No claimed tickets.", show_alert=True)
        # build menu
        buttons = [
            [InlineKeyboardButton(text=tid, callback_data=f"switch:{tid}")]
            for tid in session["claimed"]
        ]
        buttons.append(
            [InlineKeyboardButton(text="✖️ Cancel", callback_data="switch:cancel")]
        )
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
        await bot.send_message(
            admin_id,
            "🔄 *Select ticket to switch to:*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb,
        )
        return await call.answer()

    # cancel menu
    if key == "cancel":
        await call.answer("Cancelled.")
        return

    # actual switch
    new_tid = key
    if new_tid not in session["claimed"]:
        return await call.answer("❌ Invalid ticket.", show_alert=True)
    session["current"] = new_tid
    # send the card for the newly active ticket
    await send_claimed_ticket_card(admin_id, new_tid)
    await call.answer(f"Switched to {new_tid}")


# ─── Handle Admin Free-Text (forward to current ticket) ────────────────────


@dp.message()
async def handle_admin_chat(message: Message):
    admin_id = message.from_user.id
    # ignore commands
    if not message.text or message.text.startswith("/"):
        return
    session = active_conversations.get(admin_id, {})
    tid = session.get("current")
    if not tid:
        # no active ticket selected
        return await message.reply(
            "ℹ️ *No ticket selected.*\nUse 🔄 Switch to pick one of your claimed tickets.",
            parse_mode=ParseMode.MARKDOWN,
        )

    # forward to user-bot
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
        try:
            resp = await sess.post(
                USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": message.text}
            )
            logger.info(
                "POST %s → ticket %s returned %d", USER_BOT_REPLY_URL, tid, resp.status
            )
            if resp.status != 200:
                logger.warning("Admin-Bot /reply returned %d", resp.status)
            if resp.status == 404:
                # re-claim then retry
                await sess.post(
                    USER_BOT_CLAIM_URL,
                    json={"ticket_id": tid, "user_id": tickets[tid]["user_id"]},
                )
                await sess.post(
                    USER_BOT_REPLY_URL, json={"ticket_id": tid, "text": message.text}
                )
        except Exception:
            logger.exception("Failed to forward to User-Bot for ticket %s", tid)


# ─── Set Up Bot Commands & Polling ──────────────────────────────────────────


async def setup_bot_commands():
    commands = [
        BotCommand(command="help", description="Show available commands"),
        BotCommand(command="browse", description="Browse all tickets"),
        BotCommand(command="open", description="List open tickets"),
        BotCommand(command="pending", description="List pending tickets"),
        BotCommand(command="closed", description="List closed tickets"),
        BotCommand(command="status", description="Check system status"),
        BotCommand(command="auth", description="Authenticate with a token"),
    ]
    if SUPERUSER_ID:
        commands.append(
            BotCommand(
                command="issue_token", description="Create admin token (superuser only)"
            )
        )
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())


@dp.message(Command(commands=["help", "start"]))
async def cmd_help(message: Message):
    user_id = message.from_user.id
    is_admin_user = is_admin(user_id)
    is_super_user = is_superuser(user_id)
    help_lines = ["👋 *Welcome to the Admin Support Bot*", ""]
    if not is_admin_user:
        help_lines += [
            "*Authentication:*",
            "• /auth <token> - Authenticate as admin",
            "",
            "Contact your superuser for a token.",
        ]
    else:
        help_lines += [
            "*Ticket Commands:*",
            "• /browse - Browse open tickets",
            "• /open   - List open tickets",
            "• /pending - List in-progress tickets",
            "• /closed - List closed tickets",
            "• After claiming, use the buttons to Close, Suspend, or Switch between your claimed tickets.",
            "• Type your message normally to reply to the *currently selected* ticket.",
        ]
        if is_super_user:
            help_lines += [
                "",
                "*Superuser Commands:*",
                "• /issue_token - Generate new admin token",
            ]
    await message.reply("\n".join(help_lines), parse_mode=ParseMode.MARKDOWN)


async def main():
    await setup_bot_commands()
    task = asyncio.create_task(dp.start_polling(bot, skip_updates=True))
    try:
        while True:
            if pending_notifications:
                note = pending_notifications.pop(0)
                await bot.send_message(
                    chat_id=note["chat_id"],
                    text=note["text"],
                    parse_mode=ParseMode.MARKDOWN,
                )
            await asyncio.sleep(1)
    finally:
        await task


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
