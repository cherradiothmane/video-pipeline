"""
Unit and integration tests for the video pipeline.

Run:
    pytest tests/ -v
    pytest tests/test_transcriber.py -v  (single module)
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Transcriber tests ────────────────────────────────────────────────────────

class TestTranscriptionSegment:
    def test_srt_format(self):
        from pipeline.transcriber import TranscriptionSegment
        seg = TranscriptionSegment(start=65.5, end=72.123, text="Hello world")
        srt = seg.to_srt_block(1)
        assert "00:01:05,500 --> 00:01:12,123" in srt
        assert "Hello world" in srt
        assert "1\n" in srt

    def test_srt_with_hours(self):
        from pipeline.transcriber import TranscriptionSegment
        seg = TranscriptionSegment(start=3661.0, end=3665.0, text="Test")
        srt = seg.to_srt_block(2)
        assert "01:01:01,000 --> 01:01:05,000" in srt


class TestTranscriptionResult:
    def test_full_text(self):
        from pipeline.transcriber import TranscriptionResult, TranscriptionSegment
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0, end=2, text="Hello"),
                TranscriptionSegment(start=2, end=4, text="world"),
            ],
            language="en",
            language_probability=0.99,
            duration=4.0,
            full_text="Hello world",
        )
        assert result.full_text == "Hello world"
        assert len(result.segments) == 2

    def test_to_srt(self):
        from pipeline.transcriber import TranscriptionResult, TranscriptionSegment
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0, end=2, text="First segment"),
                TranscriptionSegment(start=2, end=5, text="Second segment"),
            ],
            language="en",
            language_probability=0.99,
            duration=5.0,
            full_text="First segment Second segment",
        )
        srt = result.to_srt()
        assert "1\n" in srt
        assert "2\n" in srt
        assert "First segment" in srt
        assert "Second segment" in srt

    def test_segments_in_range(self):
        from pipeline.transcriber import TranscriptionResult, TranscriptionSegment
        result = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0, end=5, text="A"),
                TranscriptionSegment(start=10, end=15, text="B"),
                TranscriptionSegment(start=20, end=25, text="C"),
            ],
            language="en",
            language_probability=0.9,
            duration=25.0,
            full_text="A B C",
        )
        in_range = result.get_segments_in_range(8, 16)
        assert len(in_range) == 1
        assert in_range[0].text == "B"


# ─── Summarizer tests ─────────────────────────────────────────────────────────

class TestSummaryResult:
    def test_to_markdown(self):
        from pipeline.summarizer import SummaryResult, ChapterSummary
        result = SummaryResult(
            title="Test Video",
            summary="This is a test summary.",
            description="Short description.",
            category="Tutoriel",
            keywords=["python", "AI", "video"],
            chapters=[
                ChapterSummary(start=0, end=60, title="Introduction", description="Intro section.")
            ],
            language="fr",
        )
        md = result.to_markdown()
        assert "# Test Video" in md
        assert "Tutoriel" in md
        assert "python" in md
        assert "Introduction" in md
        assert "00:01" in md  # end timecode


class TestSummarizer:
    def test_parse_clean_json(self):
        from pipeline.summarizer import Summarizer
        s = Summarizer.__new__(Summarizer)
        data = s._parse_json_response('{"title": "Hello", "summary": "World", "description": "", "category": "Test", "keywords": [], "chapters": []}')
        assert data["title"] == "Hello"

    def test_parse_json_with_fences(self):
        from pipeline.summarizer import Summarizer
        s = Summarizer.__new__(Summarizer)
        raw = '```json\n{"title": "Hi", "summary": "S", "description": "D", "category": "C", "keywords": [], "chapters": []}\n```'
        data = s._parse_json_response(raw)
        assert data["title"] == "Hi"

    def test_parse_invalid_json_raises(self):
        from pipeline.summarizer import Summarizer
        s = Summarizer.__new__(Summarizer)
        with pytest.raises((ValueError, Exception)):
            s._parse_json_response("not json at all {{broken")


# ─── OCR tests ────────────────────────────────────────────────────────────────

class TestOCREntry:
    def test_text_cleanup(self):
        from pipeline.ocr_extractor import OCREntry
        entry = OCREntry(timecode=5.0, frame_index=150, text="  hello   world  ", detections=[])
        assert entry.text == "hello world"


class TestOCRResult:
    def _make_result(self):
        from pipeline.ocr_extractor import OCRResult, OCREntry
        return OCRResult(
            entries=[
                OCREntry(timecode=1.0, frame_index=30, text="TITLE SCREEN"),
                OCREntry(timecode=5.0, frame_index=150, text="Chapter 1"),
                OCREntry(timecode=10.0, frame_index=300, text="TITLE SCREEN"),  # duplicate
            ],
            total_frames_sampled=30,
            frames_with_text=3,
            video_path="/fake/video.mp4",
        )

    def test_get_text_at(self):
        result = self._make_result()
        entries = result.get_text_at(5.0, window=1.0)
        assert any(e.text == "Chapter 1" for e in entries)

    def test_get_unique_texts(self):
        result = self._make_result()
        unique = result.get_unique_texts()
        assert len(unique) == 2  # TITLE SCREEN deduplicated

    def test_to_timeline(self):
        result = self._make_result()
        timeline = result.to_timeline()
        assert timeline[0]["timecode"] < timeline[1]["timecode"]
        assert "timecode_fmt" in timeline[0]


# ─── Pipeline integration test (mocked) ──────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_success(self, tmp_path):
        from pipeline.pipeline import VideoPipeline, PipelineConfig
        from pipeline.transcriber import TranscriptionResult, TranscriptionSegment
        from pipeline.summarizer import SummaryResult
        from pipeline.ocr_extractor import OCRResult

        config = PipelineConfig(output_dir=str(tmp_path), ocr_enabled=True)
        pipeline = VideoPipeline(config)

        # Mock all sub-components
        mock_transcription = TranscriptionResult(
            segments=[TranscriptionSegment(start=0, end=5, text="Test content")],
            language="fr",
            language_probability=0.95,
            duration=60.0,
            full_text="Test content",
        )
        mock_summary = SummaryResult(
            title="Test Title",
            summary="Test summary.",
            description="Test description.",
            category="Test",
            keywords=["test"],
            chapters=[],
            language="fr",
        )
        mock_ocr = OCRResult(entries=[], total_frames_sampled=60, frames_with_text=0, video_path="/fake")

        pipeline.transcriber = MagicMock()
        pipeline.transcriber.transcribe.return_value = mock_transcription

        pipeline.summarizer = MagicMock()
        pipeline.summarizer.summarize.return_value = mock_summary

        pipeline.ocr_extractor = MagicMock()
        pipeline.ocr_extractor.extract.return_value = mock_ocr

        pipeline._initialized = True

        # Create a fake video file
        fake_video = tmp_path / "test_video.mp4"
        fake_video.write_bytes(b"fake video data")

        result = pipeline.process_video(str(fake_video))

        assert result.status == "done"
        assert result.video_name == "test_video"
        assert result.transcription == mock_transcription
        assert result.summary == mock_summary
        assert result.ocr == mock_ocr

        # Check outputs were saved
        assert (tmp_path / "test_video" / "result.json").exists()
        assert (tmp_path / "test_video" / "test_video.srt").exists()
        assert (tmp_path / "test_video" / "summary.md").exists()

    def test_pipeline_handles_failure(self, tmp_path):
        from pipeline.pipeline import VideoPipeline, PipelineConfig

        config = PipelineConfig(output_dir=str(tmp_path))
        pipeline = VideoPipeline(config)

        pipeline.transcriber = MagicMock()
        pipeline.transcriber.transcribe.side_effect = RuntimeError("CUDA out of memory")
        pipeline._initialized = True

        fake_video = tmp_path / "bad_video.mp4"
        fake_video.write_bytes(b"fake")

        result = pipeline.process_video(str(fake_video))

        assert result.status == "failed"
        assert "CUDA out of memory" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
