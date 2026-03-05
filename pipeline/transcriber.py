"""
Audio transcription using faster-whisper (CTranslate2-optimized Whisper).
Produces timestamped segments with word-level timecodes.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    start: float        # seconds
    end: float          # seconds
    text: str
    words: list = field(default_factory=list)  # word-level timecodes if available
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    def to_srt_block(self, index: int) -> str:
        def fmt(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int((s - int(s)) * 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        return f"{index}\n{fmt(self.start)} --> {fmt(self.end)}\n{self.text.strip()}\n"


@dataclass
class TranscriptionResult:
    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    duration: float
    full_text: str

    def to_srt(self) -> str:
        blocks = [seg.to_srt_block(i + 1) for i, seg in enumerate(self.segments)]
        return "\n".join(blocks)

    def to_plain_text(self) -> str:
        return self.full_text

    def get_segments_in_range(self, start: float, end: float) -> list[TranscriptionSegment]:
        return [s for s in self.segments if s.start >= start and s.end <= end]


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str = "./models",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                )
            logger.info(
                f"Loading Whisper model '{self.model_size}' on {self.device} ({self.compute_type})"
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
            logger.info("Whisper model loaded.")

    def transcribe(
        self,
        audio_or_video_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file.
        
        Args:
            audio_or_video_path: Path to audio or video file (ffmpeg handles extraction)
            language: ISO-639-1 language code (e.g. "fr", "en") or None for auto-detect
            beam_size: Beam search size (higher = more accurate but slower)
            vad_filter: Filter silence using Voice Activity Detection
            word_timestamps: Enable word-level timestamps
            
        Returns:
            TranscriptionResult with segments and metadata
        """
        self._load_model()

        logger.info(f"Transcribing: {audio_or_video_path}")

        segments_gen, info = self._model.transcribe(
            audio_or_video_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        logger.info(
            f"Detected language: {info.language} (prob={info.language_probability:.2f}), "
            f"duration: {info.duration:.1f}s"
        )

        segments = []
        full_text_parts = []

        for seg in segments_gen:
            words = []
            if word_timestamps and seg.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                    for w in seg.words
                ]

            segment = TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
            )
            segments.append(segment)
            full_text_parts.append(seg.text.strip())

            logger.debug(f"  [{seg.start:.2f}s → {seg.end:.2f}s] {seg.text.strip()}")

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            full_text=full_text,
        )
