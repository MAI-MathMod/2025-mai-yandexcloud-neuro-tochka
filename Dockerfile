# Use official Python runtime
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY . .

# Expose both ports
EXPOSE 8080 8096

# Decide at runtime which bot to launch
# Pass SERVICE=admin or SERVICE=user as a build-arg or env var
ARG SERVICE
ENV SERVICE=${SERVICE}

# Start Flask + Aiogram for the chosen service
# admin → listens on 8096; user → listens on 8080
CMD ["sh", "-c", "\
    if [ \"$SERVICE\" = \"admin\" ]; then \
    python admin_bot.py; \
    else \
    python user_bot.py; \
    fi \
    "]
