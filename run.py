#!/usr/bin/env python3
"""
Command-line interface for the video pipeline.

Usage:
    # Process a single video
    python run.py process video.mp4

    # Process a batch
    python run.py batch /path/to/videos/*.mp4

    # Start the API server
    python run.py server

    # Check system status
    python run.py status
"""
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def cmd_process(args):
    from pipeline import PipelineConfig, VideoPipeline

    config = PipelineConfig(
        whisper_model=args.whisper_model,
        whisper_device=args.device,
        ollama_model=args.llm_model,
        ollama_host=args.ollama_host,
        ocr_enabled=not args.no_ocr,
        whisper_language=args.language,
        output_dir=args.output_dir,
    )

    pipeline = VideoPipeline(config)

    def progress(step, pct):
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {step:<20}", end="", flush=True)
        if pct == 100:
            print()

    print(f"\n🎬 Processing: {args.video}")
    result = pipeline.process_video(args.video, progress_callback=progress)

    if result.status == "done":
        print(f"\n✅ Done in {result.processing_time_seconds:.1f}s")
        if result.summary:
            print(f"\n📌 Title   : {result.summary.title}")
            print(f"🏷️  Category : {result.summary.category}")
            print(f"🔑 Keywords : {', '.join(result.summary.keywords)}")
            print(f"\n📝 Summary :\n{result.summary.summary}")
        if result.ocr and result.ocr.frames_with_text > 0:
            print(f"\n🔍 OCR: {result.ocr.frames_with_text} frames with text")
        print(f"\n💾 Output saved to: {args.output_dir}/{result.video_name}/")
    else:
        print(f"\n❌ Failed: {result.error}")
        sys.exit(1)


def cmd_batch(args):
    from pipeline import PipelineConfig, VideoPipeline

    # Expand globs
    paths = []
    for pattern in args.videos:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif Path(pattern).exists():
            paths.append(pattern)
        else:
            print(f"⚠️  No files matched: {pattern}")

    if not paths:
        print("❌ No video files found.")
        sys.exit(1)

    print(f"\n📦 Batch processing {len(paths)} videos...")
    for i, p in enumerate(paths):
        print(f"  {i+1}. {p}")

    config = PipelineConfig(
        whisper_model=args.whisper_model,
        whisper_device=args.device,
        ollama_model=args.llm_model,
        ollama_host=args.ollama_host,
        ocr_enabled=not args.no_ocr,
        output_dir=args.output_dir,
    )

    pipeline = VideoPipeline(config)

    def progress(video_name, step, pct):
        print(f"  [{video_name}] {step}: {pct}%")

    results = pipeline.process_batch(paths, progress_callback=progress)

    print("\n📊 Batch Results:")
    for r in results:
        status_icon = "✅" if r.status == "done" else "❌"
        print(f"  {status_icon} {r.video_name}: {r.status} ({r.processing_time_seconds:.1f}s)")
        if r.error:
            print(f"      Error: {r.error}")


def cmd_server(args):
    import uvicorn
    print(f"\n🚀 Starting API server on http://0.0.0.0:{args.port}")
    os.environ.setdefault("WHISPER_MODEL", args.whisper_model)
    os.environ.setdefault("OLLAMA_MODEL", args.llm_model)
    os.environ.setdefault("OLLAMA_HOST", args.ollama_host)
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        workers=1,
    )


def cmd_status(args):
    import requests
    print("\n🔍 Checking system status...\n")

    # Check Ollama
    try:
        resp = requests.get(f"{args.ollama_host}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"✅ Ollama running at {args.ollama_host}")
        print(f"   Models: {', '.join(models) if models else 'none'}")
    except Exception as e:
        print(f"❌ Ollama not reachable: {e}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu} ({vram:.1f} GB VRAM)")
        else:
            print("⚠️  No CUDA GPU detected (will use CPU)")
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")

    # Check packages
    packages = {
        "faster-whisper": "faster_whisper",
        "easyocr": "easyocr",
        "opencv-python": "cv2",
        "fastapi": "fastapi",
        "celery": "celery",
    }
    print("\n📦 Package availability:")
    for name, module in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (not installed)")


def main():
    parser = argparse.ArgumentParser(description="Video Pipeline CLI")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--llm-model", default="qwen2.5:32b", help="Ollama model name")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--language", default=None, help="Language code for Whisper (e.g. fr, en)")

    subparsers = parser.add_subparsers(dest="command")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a single video")
    process_parser.add_argument("video", help="Path to video file")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple videos")
    batch_parser.add_argument("videos", nargs="+", help="Video paths or glob patterns")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start REST API server")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--reload", action="store_true")

    # Status command
    subparsers.add_parser("status", help="Check system status")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "server":
        cmd_server(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
