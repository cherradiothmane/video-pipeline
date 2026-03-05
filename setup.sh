#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# VideoPipeline Setup Script
# Run once after cloning: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[SETUP]${NC} $1"; }
ok()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERR]${NC} $1"; }

log "VideoPipeline Setup"
echo ""

# ─── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Install Python 3.11+"
    exit 1
fi
PYVER=$(python3 --version)
ok "Python: $PYVER"

# ─── Check CUDA ──────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    ok "GPU detected: $GPU"
else
    warn "nvidia-smi not found. GPU acceleration may not be available."
fi

# ─── Check ffmpeg ────────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    err "ffmpeg not found! Install with: sudo apt install ffmpeg"
    exit 1
fi

# ─── Python venv ─────────────────────────────────────────────────────────────
log "Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Virtual environment created at .venv/"
else
    ok "Virtual environment already exists."
fi

source .venv/bin/activate

# ─── pip install ─────────────────────────────────────────────────────────────
log "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Python packages installed."

# ─── .env ────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    ok ".env created from .env.example — edit it to set your model preferences."
else
    ok ".env already exists."
fi

# ─── Directories ─────────────────────────────────────────────────────────────
mkdir -p uploads outputs models
ok "Directories created: uploads/ outputs/ models/"

# ─── Ollama check ────────────────────────────────────────────────────────────
echo ""
log "Checking Ollama..."
if command -v ollama &>/dev/null; then
    ok "Ollama is installed."
    
    # Check if model is pulled
    LLM_MODEL=${OLLAMA_MODEL:-qwen2.5:32b}
    if ollama list 2>/dev/null | grep -q "$LLM_MODEL"; then
        ok "Model '$LLM_MODEL' is available."
    else
        warn "Model '$LLM_MODEL' not found."
        echo ""
        echo "  Pull it with:"
        echo "    ollama pull $LLM_MODEL"
        echo ""
        echo "  Or choose a lighter model for testing:"
        echo "    ollama pull llama3.1:8b"
        echo "    ollama pull mistral:7b"
    fi
else
    warn "Ollama not installed."
    echo ""
    echo "  Install with:"
    echo "    curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then pull your model:"
    echo "    ollama pull qwen2.5:32b"
fi

# ─── UI setup ────────────────────────────────────────────────────────────────
echo ""
log "Setting up React UI..."
if command -v node &>/dev/null; then
    ok "Node.js: $(node --version)"
    cd ui && npm install --quiet && cd ..
    ok "UI dependencies installed. Build with: cd ui && npm run build"
else
    warn "Node.js not found. UI setup skipped."
    echo "  Install from: https://nodejs.org/"
fi

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""
ok "Setup complete!"
echo ""
echo "  Quick start:"
echo ""
echo "  1. Start Ollama:        ollama serve"
echo "  2. Check status:        python run.py status"
echo "  3. Process a video:     python run.py process myvideo.mp4"
echo "  4. Start API server:    python run.py server"
echo "  5. Launch UI:           cd ui && npm run dev"
echo ""
echo "  Or with Docker:"
echo "    docker-compose up -d"
echo ""
