"""
Main pipeline orchestrator - chains transcription, LLM summarization, and OCR
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

from .transcriber import Transcriber, TranscriptionResult
from .summarizer import Summarizer, SummaryResult
from .ocr_extractor import OCRExtractor, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Whisper settings
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_language: Optional[str] = None  # None = auto-detect

    # LLM settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:32b"
    llm_language: str = "fr"  # Output language for summaries

    # OCR settings
    ocr_enabled: bool = True
    ocr_languages: list = field(default_factory=lambda: ["fr", "en"])
    ocr_fps_sample: float = 1.0  # Frames per second to sample
    ocr_min_confidence: float = 0.5

    # Output settings
    output_dir: str = "./outputs"
    save_srt: bool = True
    save_json: bool = True


@dataclass
class PipelineResult:
    video_path: str
    video_name: str
    duration_seconds: float
    processing_time_seconds: float
    transcription: Optional[TranscriptionResult] = None
    summary: Optional[SummaryResult] = None
    ocr: Optional[OCRResult] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, processing, done, failed

    def to_dict(self):
        d = asdict(self)
        return d


class VideoPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.transcriber = None
        self.summarizer = None
        self.ocr_extractor = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of all models."""
        logger.info("Initializing pipeline models...")

        logger.info("Loading Whisper model...")
        self.transcriber = Transcriber(
            model_size=self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
        )

        logger.info("Connecting to Ollama LLM...")
        self.summarizer = Summarizer(
            host=self.config.ollama_host,
            model=self.config.ollama_model,
            language=self.config.llm_language,
        )

        if self.config.ocr_enabled:
            logger.info("Loading OCR model...")
            self.ocr_extractor = OCRExtractor(
                languages=self.config.ocr_languages,
                gpu=True,
                min_confidence=self.config.ocr_min_confidence,
            )

        self._initialized = True
        logger.info("Pipeline initialized successfully.")

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> PipelineResult:
        """
        Process a single video through the full pipeline.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback(step_name, percent) for progress updates
        """
        if not self._initialized:
            self.initialize()

        video_path = str(Path(video_path).resolve())
        video_name = Path(video_path).stem
        start_time = time.time()

        result = PipelineResult(
            video_path=video_path,
            video_name=video_name,
            duration_seconds=0.0,
            processing_time_seconds=0.0,
            status="processing",
        )

        def _progress(step: str, pct: int):
            logger.info(f"[{video_name}] {step}: {pct}%")
            if progress_callback:
                progress_callback(step, pct)

        try:
            # Step 1: Transcription
            _progress("transcription", 0)
            logger.info(f"Transcribing: {video_path}")
            result.transcription = self.transcriber.transcribe(
                video_path,
                language=self.config.whisper_language,
            )
            result.duration_seconds = result.transcription.duration
            _progress("transcription", 100)

            # Step 2: LLM Summarization
            _progress("summarization", 0)
            logger.info(f"Summarizing transcript...")
            result.summary = self.summarizer.summarize(result.transcription)
            _progress("summarization", 100)

            # Step 3: OCR
            if self.config.ocr_enabled and self.ocr_extractor:
                _progress("ocr", 0)
                logger.info(f"Extracting text from frames...")
                result.ocr = self.ocr_extractor.extract(
                    video_path,
                    fps_sample=self.config.ocr_fps_sample,
                    progress_callback=lambda p: _progress("ocr", p),
                )
                _progress("ocr", 100)

            result.status = "done"
            result.processing_time_seconds = time.time() - start_time

            # Save outputs
            self._save_outputs(result)

            logger.info(
                f"Pipeline complete for '{video_name}' in {result.processing_time_seconds:.1f}s"
            )

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.processing_time_seconds = time.time() - start_time
            logger.error(f"Pipeline failed for '{video_name}': {e}", exc_info=True)

        return result

    def process_batch(
        self,
        video_paths: list[str],
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> list[PipelineResult]:
        """
        Process multiple videos sequentially.
        
        Args:
            video_paths: List of video file paths
            progress_callback: Optional callback(video_name, step, percent)
        """
        if not self._initialized:
            self.initialize()

        results = []
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            video_name = Path(video_path).stem

            def _cb(step, pct):
                if progress_callback:
                    progress_callback(video_name, step, pct)

            result = self.process_video(video_path, progress_callback=_cb)
            results.append(result)

        return results

    def _save_outputs(self, result: PipelineResult):
        """Save results to disk."""
        output_dir = Path(self.config.output_dir) / result.video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full JSON result
        if self.config.save_json:
            json_path = output_dir / "result.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON: {json_path}")

        # Save SRT subtitle file
        if self.config.save_srt and result.transcription:
            srt_path = output_dir / f"{result.video_name}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(result.transcription.to_srt())
            logger.info(f"Saved SRT: {srt_path}")

        # Save summary as markdown
        if result.summary:
            md_path = output_dir / "summary.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(result.summary.to_markdown())
            logger.info(f"Saved summary: {md_path}")

        # Save OCR results
        if result.ocr and result.ocr.entries:
            ocr_path = output_dir / "ocr_results.json"
            with open(ocr_path, "w", encoding="utf-8") as f:
                json.dump(
                    [e.__dict__ for e in result.ocr.entries],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Saved OCR: {ocr_path}")
