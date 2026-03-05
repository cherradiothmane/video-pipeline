# 🎬 VideoPipeline

Pipeline offline complet pour la transcription, catégorisation et extraction OCR de vidéos. Conçu pour tourner sur serveur GPU local avec un RTX Ada 48 Go VRAM.

## Architecture

```
video-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py          # Orchestrateur principal
│   ├── transcriber.py       # Whisper STT (faster-whisper)
│   ├── summarizer.py        # LLM résumé/catégorie via Ollama
│   └── ocr_extractor.py     # OCR sur frames (EasyOCR)
├── api/
│   ├── server.py            # FastAPI REST + WebSocket
│   └── worker.py            # Celery worker (batch async)
├── ui/
│   ├── src/App.jsx          # Dashboard React
│   ├── src/main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── config/
│   └── settings.py          # Config depuis variables d'env
├── tests/
│   └── test_pipeline.py     # Tests unitaires
├── run.py                   # CLI principal
├── setup.sh                 # Script d'installation
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Stack technique

| Composant | Outil | VRAM |
|---|---|---|
| Transcription STT | `faster-whisper` large-v3 | ~3 Go |
| LLM résumé | `Ollama` + Qwen2.5-32B | ~20 Go |
| OCR | `EasyOCR` (GPU) | ~2 Go |
| **Total simultané** | | **~25 Go** |

## Installation rapide

```bash
git clone <repo>
cd video-pipeline
bash setup.sh
```

### Prérequis système
- Python 3.11+
- CUDA 12+
- ffmpeg (`sudo apt install ffmpeg`)
- Node.js 18+ (pour l'interface web)

### 1. Installer Ollama et le modèle LLM

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:32b        # 20 Go VRAM — recommandé
# ou
ollama pull mistral-small3.1:24b   # alternative
# ou pour tester rapidement
ollama pull llama3.1:8b            # léger, 5 Go VRAM
```

### 2. Installer les dépendances Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurer `.env`

```bash
cp .env.example .env
# Editer si nécessaire (le défaut fonctionne avec qwen2.5:32b)
```

## Utilisation

### CLI (ligne de commande)

```bash
# Vérifier que tout est en ordre
python run.py status

# Traiter une vidéo
python run.py process ma_video.mp4

# Traiter un lot de vidéos
python run.py batch /chemin/videos/*.mp4

# Lancer le serveur API
python run.py server --port 8000

# Options disponibles
python run.py --help
python run.py process --help
```

### API REST

```bash
# Démarrer le serveur
python run.py server

# Upload et traitement
curl -X POST http://localhost:8000/jobs/upload \
  -F "file=@ma_video.mp4"

# Vérifier le statut
curl http://localhost:8000/jobs/<job_id>

# Récupérer le résultat
curl http://localhost:8000/jobs/<job_id>/result

# Télécharger le SRT
curl http://localhost:8000/jobs/<job_id>/download/ma_video.srt -O

# Lister tous les jobs
curl http://localhost:8000/jobs
```

### WebSocket (suivi temps réel)

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/<job_id>");
ws.onmessage = (e) => {
  const update = JSON.parse(e.data);
  console.log(update.step, update.progress + "%");
};
```

### Interface web

```bash
cd ui
npm install
npm run dev     # Dev: http://localhost:3000
npm run build   # Build production (servie par FastAPI sur port 8000)
```

## Docker

```bash
# Démarrer tout le stack
docker-compose up -d

# Logs
docker-compose logs -f api
docker-compose logs -f worker

# Arrêter
docker-compose down
```

Après le démarrage, pull le modèle Ollama dans le container :
```bash
docker-compose exec ollama ollama pull qwen2.5:32b
```

## Format de sortie JSON

Pour chaque vidéo traitée, le résultat est sauvegardé dans `outputs/<nom_video>/result.json` :

```json
{
  "video_path": "/path/to/video.mp4",
  "video_name": "video",
  "duration_seconds": 1234.5,
  "processing_time_seconds": 45.2,
  "status": "done",
  "transcription": {
    "language": "fr",
    "language_probability": 0.99,
    "duration": 1234.5,
    "full_text": "...",
    "segments": [
      { "start": 0.0, "end": 3.5, "text": "Bonjour et bienvenue...", "words": [...] }
    ]
  },
  "summary": {
    "title": "Introduction à Python pour les débutants",
    "summary": "Dans cette vidéo...",
    "description": "Apprenez Python en partant de zéro...",
    "category": "Tutoriel",
    "keywords": ["python", "programmation", "débutant"],
    "chapters": [
      { "start": 0, "end": 120, "title": "Introduction", "description": "...", "keywords": [...] }
    ]
  },
  "ocr": {
    "total_frames_sampled": 1234,
    "frames_with_text": 87,
    "entries": [
      { "timecode": 12.0, "frame_index": 360, "text": "SLIDE: Introduction" }
    ]
  }
}
```

Autres fichiers générés :
- `outputs/<nom>/result.json` — résultat complet
- `outputs/<nom>/<nom>.srt` — sous-titres SRT avec timecodes
- `outputs/<nom>/summary.md` — résumé en Markdown
- `outputs/<nom>/ocr_results.json` — textes OCR

## Tests

```bash
# Lancer tous les tests (pas besoin de GPU)
pytest tests/ -v

# Test rapide sans Ollama/Whisper (tout mocké)
pytest tests/test_pipeline.py -v
```

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Taille du modèle Whisper |
| `WHISPER_DEVICE` | `cuda` | `cuda` ou `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | `float16`, `int8_float16`, `int8` |
| `OLLAMA_HOST` | `http://localhost:11434` | URL du serveur Ollama |
| `OLLAMA_MODEL` | `qwen2.5:32b` | Modèle LLM à utiliser |
| `LLM_LANGUAGE` | `fr` | Langue des résumés générés |
| `OCR_ENABLED` | `true` | Activer/désactiver l'OCR |
| `OCR_FPS_SAMPLE` | `1.0` | Fréquence d'analyse des frames |
| `OCR_MIN_CONFIDENCE` | `0.5` | Seuil de confiance OCR |

## Performances estimées (RTX Ada 48 Go)

| Tâche | Temps pour 1h de vidéo |
|---|---|
| Transcription Whisper large-v3 | ~8-12 min |
| Résumé LLM (Qwen2.5-32B) | ~1-3 min |
| OCR (1 frame/sec) | ~5-10 min |
| **Total** | **~15-25 min** |

## Troubleshooting

**`CUDA out of memory`** : Réduire `WHISPER_MODEL` à `medium` ou changer `WHISPER_COMPUTE_TYPE` en `int8_float16`.

**`Ollama timeout`** : Le modèle 32B peut prendre 1-2 min à charger. Augmenter le timeout ou utiliser un modèle plus léger.

**OCR lent** : Réduire `OCR_FPS_SAMPLE` à `0.5` ou `0.25` pour analyser moins de frames.
"# video-pipeline" 
