FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (kept minimal; add build-essential if you need to compile wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

COPY . .

ENV PORT=8000
EXPOSE 8000

# app.py must have: app = Flask(__name__)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app", "--workers=2", "--threads=4", "--timeout=120"]