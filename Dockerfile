FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Optional but helps with some builds + keeps pip up to date
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

COPY . .

ENV PORT=4000
EXPOSE 4000

CMD ["gunicorn", "-b", "0.0.0.0:4000", "app:app", "--workers=2", "--threads=4", "--timeout=120"]