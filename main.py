import threading
import uuid
import asyncio
import logging
import os
from dotenv import load_dotenv
import sqlite3
from jose import jwt, JWTError
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.filters import Command
import signal
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
ADMIN_BOT_TOKEN = os.getenv("ADMIN_BOT_TOKEN")
USER_BOT_TOKEN = os.getenv("USER_BOT_TOKEN")
JWT_SECRET = os.getenv("JWT_SECRET")
TOKEN_EXPIRATION_MINUTES = int(os.getenv("TOKEN_EXPIRATION_MINUTES", "30"))

if not ADMIN_BOT_TOKEN or not USER_BOT_TOKEN or not JWT_SECRET:
    logger.error("Missing required environment variables. Please check your .env file.")
    sys.exit(1)

# Initialize Flask app and Aiogram bots
app = Flask(__name__)
admin_bot = Bot(token=ADMIN_BOT_TOKEN)
user_bot = Bot(token=USER_BOT_TOKEN)
dp = Dispatcher()

# Initialize SQLite
DB_PATH = os.getenv("DB_PATH", "admin_bot.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create a list to store pending notifications
pending_notifications = []

# Create tables
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER UNIQUE NOT NULL,
    is_superuser BOOLEAN NOT NULL DEFAULT 0
)
"""
)
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS tokens (
    token TEXT PRIMARY KEY,
    telegram_id INTEGER NOT NULL,
    salt TEXT NOT NULL,
    expires_at DATETIME NOT NULL
)
"""
)
conn.commit()

# Ensure superuser exists in admins table
SUPERUSER_ID = int(os.getenv("SUPERUSER_ID", "0"))
if SUPERUSER_ID == 0:
    logger.error("SUPERUSER_ID is not set or invalid. Please check your .env file.")
    sys.exit(1)

cursor.execute(
    "INSERT OR IGNORE INTO admins (telegram_id, is_superuser) VALUES (?, 1)",
    (SUPERUSER_ID,),
)
conn.commit()

# In-memory ticket store
tickets = {}  # ticket_id -> {user_id, message, status}


# --- External API to receive tickets ---
@app.route("/tickets", methods=["POST"])
def receive_ticket():
    data = request.json or {}
    user_id = data.get("user_id")
    message_text = data.get("message")
    if not user_id or not message_text:
        logger.warning("receive_ticket: missing fields %s", data)
        return jsonify({"error": "Missing user_id or message"}), 400

    ticket_id = str(uuid.uuid4())
    tickets[ticket_id] = {"user_id": user_id, "message": message_text, "status": "open"}
    logger.info("Ticket %s created by %s", ticket_id, user_id)

    notification_data = {
        "chat_id": SUPERUSER_ID,
        "text": f"🆕 New ticket `{ticket_id}` from {user_id}:\n{message_text}",
        "ticket_id": ticket_id,
    }
    pending_notifications.append(notification_data)

    return jsonify({"ticket_id": ticket_id}), 200


# --- Bot command handlers ---
async def cmd_issue_token(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT is_superuser FROM admins WHERE telegram_id = ?", (user_id,))
    row = cursor.fetchone()
    if not row or not row[0]:
        return await message.reply("❌ You are not allowed to issue tokens.")

    salt = uuid.uuid4().hex
    now = datetime.utcnow()
    exp = now + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    payload = {"telegram_id": user_id, "salt": salt, "iat": now, "exp": exp}
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    cursor.execute(
        "INSERT INTO tokens (token, telegram_id, salt, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, salt, exp),
    )
    conn.commit()

    await message.reply("✅ Admin token created (click the token below to copy):")
    await message.answer(f"```\n{token}\n```", parse_mode=ParseMode.MARKDOWN)


async def cmd_auth(message: Message):
    parts = message.text.strip().split()
    if len(parts) != 2:
        return await message.reply("Usage: /auth <token>")
    token = parts[1]
    cursor.execute(
        "SELECT telegram_id, salt, expires_at FROM tokens WHERE token = ?", (token,)
    )
    row = cursor.fetchone()
    if not row:
        return await message.reply("❌ Invalid or revoked token.")

    stored_tg, stored_salt, expires_at = row
    if datetime.utcnow() > datetime.fromisoformat(expires_at):
        return await message.reply("❌ Token has expired.")

    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        return await message.reply("❌ Invalid token signature.")

    if decoded.get("telegram_id") != stored_tg or decoded.get("salt") != stored_salt:
        return await message.reply("❌ Token validation failed.")

    cursor.execute(
        "INSERT OR IGNORE INTO admins (telegram_id, is_superuser) VALUES (?, 0)",
        (message.from_user.id,),
    )
    conn.commit()
    cursor.execute("DELETE FROM tokens WHERE token = ?", (token,))
    conn.commit()

    await message.reply("✅ You are now authenticated as admin.")


async def cmd_ticket(message: Message):
    user_id = message.from_user.id
    cursor.execute("SELECT 1 FROM admins WHERE telegram_id = ?", (user_id,))
    if not cursor.fetchone():
        return await message.reply("❌ You are not authorized to view tickets.")

    if not tickets:
        return await message.reply("No tickets available.")

    lines = []
    for tid, info in tickets.items():
        lines.append(
            f"🆔 *{tid}*\n👤 User: `{info['user_id']}`\n💬 Message: {info['message']}\n📌 Status: {info['status']}"
        )
    await message.reply("\n\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# Register handlers
dp.message.register(cmd_issue_token, Command(commands=["issue_token"]))
dp.message.register(cmd_auth, Command(commands=["auth"]))
dp.message.register(cmd_ticket, Command(commands=["ticket"]))


# --- Run Flask and Bot ---
async def main():
    # Start bot polling
    polling_task = asyncio.create_task(dp.start_polling(admin_bot, skip_updates=True))
    try:
        while True:
            # Process pending notifications
            if pending_notifications:
                notification = pending_notifications.pop(0)
                await admin_bot.send_message(
                    chat_id=notification["chat_id"], text=notification["text"]
                )
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Stopping polling...")
        await dp.stop_polling()

# Add a shutdown function
def shutdown_app(signal_received=None, frame=None):
    """Handle shutdown gracefully"""
    logger.info("Shutdown signal received. Cleaning up...")
    
    # Close database connection
    if 'conn' in globals() and conn:
        try:
            logger.info("Closing database connection...")
            conn.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    logger.info("Forcing process termination...")
    # Force exit all threads
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_app)  # Ctrl+C
    signal.signal(signal.SIGTERM, shutdown_app)  # Termination signal
    
    # Run Flask without reloader to allow clean shutdown
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8096, use_reloader=False),
        daemon=True,
    )
    flask_thread.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main thread.")
        # Let the signal handler do the cleanup
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        shutdown_app()
