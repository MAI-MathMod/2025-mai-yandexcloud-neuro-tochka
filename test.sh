#!/usr/bin/env bash

# Admin Bot Flask endpoint
ADMIN_BOT_URL="http://localhost:8096"

# Dummy payload values
USER_ID=42
MESSAGE="Hello Admin, I need assistance with my account."
DIALOG='[
  {"from":"user","text":"Hello!"},
  {"from":"bot","text":"How can I help you today?"}
]'

echo "Sending dummy ticket to $ADMIN_BOT_URL/tickets..."

# With full dialog array included
curl -v -X POST "$ADMIN_BOT_URL/tickets" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": '"$USER_ID"',
    "message": "'"$MESSAGE"'",
    "dialog": '"$DIALOG"'
  }'

echo
