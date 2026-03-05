# ─── Build stage ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies (ffmpeg required by faster-whisper)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python python3.11 1

WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── App code ─────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p uploads outputs models

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
