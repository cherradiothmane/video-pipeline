"""
OCR extraction from video frames using EasyOCR with GPU acceleration.
Samples frames at configurable intervals and extracts visible text with timecodes.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _convert_numpy(obj):
    """Recursively convert numpy int64/float32 to native Python types."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    return obj


@dataclass
class OCREntry:
    timecode: float          # seconds
    frame_index: int
    text: str                # concatenated text from frame
    detections: list = field(default_factory=list)  # raw bboxes + confidence

    def __post_init__(self):
        # Clean up text
        self.text = " ".join(self.text.split())
        # Convert numpy types to native Python for JSON serialization
        self.detections = _convert_numpy(self.detections)


@dataclass
class OCRResult:
    entries: list[OCREntry]
    total_frames_sampled: int
    frames_with_text: int
    video_path: str

    def get_text_at(self, timecode: float, window: float = 1.0) -> list[OCREntry]:
        """Get OCR entries near a specific timecode."""
        return [e for e in self.entries if abs(e.timecode - timecode) <= window]

    def get_unique_texts(self) -> list[str]:
        """Deduplicated list of all texts found."""
        seen = set()
        unique = []
        for entry in self.entries:
            if entry.text and entry.text not in seen:
                seen.add(entry.text)
                unique.append(entry.text)
        return unique

    def to_timeline(self) -> list[dict]:
        """Sorted timeline of text appearances."""
        return [
            {
                "timecode": e.timecode,
                "timecode_fmt": f"{int(e.timecode//60):02d}:{int(e.timecode%60):02d}",
                "text": e.text,
            }
            for e in sorted(self.entries, key=lambda x: x.timecode)
        ]


class OCRExtractor:
    def __init__(
        self,
        languages: list[str] = None,
        gpu: bool = True,
        min_confidence: float = 0.5,
        backend: str = "easyocr",  # "easyocr" or "paddleocr"
    ):
        self.languages = languages or ["fr", "en"]
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.backend = backend
        self._reader = None
        self._reader2 = None  # Secondary reader for Latin languages

    def _load_reader(self):
        if self._reader is None:
            if self.backend == "easyocr":
                try:
                    import easyocr
                except ImportError:
                    raise ImportError("easyocr not installed. Run: pip install easyocr")

                has_arabic = "ar" in self.languages
                other_langs = [l for l in self.languages if l != "ar"]

                if has_arabic and other_langs and other_langs != ["en"]:
                    # EasyOCR: Arabic only compatible with English.
                    # Use two readers: ar+en and fr+en
                    logger.info(f"Loading EasyOCR reader 1 (Arabic): [ar, en], gpu={self.gpu}")
                    self._reader = easyocr.Reader(["ar", "en"], gpu=self.gpu)
                    logger.info(f"Loading EasyOCR reader 2 (Latin): {other_langs}, gpu={self.gpu}")
                    self._reader2 = easyocr.Reader(other_langs, gpu=self.gpu)
                else:
                    langs = ["ar", "en"] if has_arabic else self.languages
                    logger.info(f"Loading EasyOCR for languages={langs}, gpu={self.gpu}")
                    self._reader = easyocr.Reader(langs, gpu=self.gpu)
                    self._reader2 = None
                logger.info("EasyOCR loaded.")

            elif self.backend == "paddleocr":
                try:
                    from paddleocr import PaddleOCR
                except ImportError:
                    raise ImportError("paddleocr not installed. Run: pip install paddleocr")

                lang = self.languages[0] if self.languages else "en"
                self._reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.gpu)
                logger.info("PaddleOCR loaded.")

    def _read_frame(self, frame):
        """Run OCR on a single frame, return (text, detections)."""
        if self.backend == "easyocr":
            detections = []
            texts = []
            for bbox, text, conf in self._reader.readtext(frame):
                if conf >= self.min_confidence and text.strip():
                    detections.append({"bbox": bbox, "text": text, "confidence": float(conf)})
                    texts.append(text.strip())
            if self._reader2 is not None:
                seen = set(texts)
                for bbox, text, conf in self._reader2.readtext(frame):
                    if conf >= self.min_confidence and text.strip() and text.strip() not in seen:
                        detections.append({"bbox": bbox, "text": text, "confidence": float(conf)})
                        texts.append(text.strip())
                        seen.add(text.strip())
            return " | ".join(texts), detections

        elif self.backend == "paddleocr":
            results = self._reader.ocr(frame, cls=True)
            detections = []
            texts = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line
                    if conf >= self.min_confidence and text.strip():
                        detections.append({"bbox": bbox, "text": text, "confidence": float(conf)})
                        texts.append(text.strip())
            return " | ".join(texts), detections

        return "", []

    def extract(
        self,
        video_path: str,
        fps_sample: float = 1.0,
        skip_similar: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> OCRResult:
        """
        Extract text from video frames via OCR.
        
        Args:
            video_path: Path to video file
            fps_sample: How many frames per second to sample (1.0 = one per second)
            skip_similar: Skip frames where text is identical to previous frame
            progress_callback: Optional callback(percent_int)
            
        Returns:
            OCRResult with all detected text entries
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python not installed. Run: pip install opencv-python")

        self._load_reader()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Calculate frame interval to match desired fps_sample
        frame_interval = max(1, int(video_fps / fps_sample))

        logger.info(
            f"Video: {total_frames} frames @ {video_fps:.1f}fps, "
            f"duration={duration:.1f}s, sampling every {frame_interval} frames"
        )

        entries = []
        frame_idx = 0
        sampled = 0
        frames_with_text = 0
        prev_text = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timecode = frame_idx / video_fps
                text, detections = self._read_frame(frame)

                if text:
                    # Skip if identical to previous (static text/watermarks)
                    if skip_similar and text == prev_text:
                        pass
                    else:
                        entries.append(
                            OCREntry(
                                timecode=timecode,
                                frame_index=frame_idx,
                                text=text,
                                detections=detections,
                            )
                        )
                        frames_with_text += 1
                        prev_text = text

                sampled += 1

                # Progress callback
                if progress_callback and total_frames > 0:
                    pct = int((frame_idx / total_frames) * 100)
                    progress_callback(pct)

            frame_idx += 1

        cap.release()

        logger.info(
            f"OCR complete: {sampled} frames sampled, {frames_with_text} with text detected."
        )

        return OCRResult(
            entries=entries,
            total_frames_sampled=sampled,
            frames_with_text=frames_with_text,
            video_path=video_path,
        )
