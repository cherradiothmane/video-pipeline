"""
LLM-based summarization, titling and categorization via Ollama (local inference).
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class ChapterSummary:
    start: float
    end: float
    title: str
    description: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class SummaryResult:
    title: str
    summary: str
    description: str
    category: str
    keywords: list[str]
    chapters: list[ChapterSummary]
    language: str

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"**Catégorie:** {self.category}",
            f"**Mots-clés:** {', '.join(self.keywords)}",
            "",
            "## Description",
            self.description,
            "",
            "## Résumé",
            self.summary,
            "",
        ]

        if self.chapters:
            lines.append("## Chapitres")
            for ch in self.chapters:
                def fmt(s):
                    m = int(s // 60)
                    sec = int(s % 60)
                    return f"{m:02d}:{sec:02d}"

                lines.append(f"### [{fmt(ch.start)} → {fmt(ch.end)}] {ch.title}")
                lines.append(ch.description)
                if ch.keywords:
                    lines.append(f"*{', '.join(ch.keywords)}*")
                lines.append("")

        return "\n".join(lines)


PROMPT_TEMPLATE = """Tu es un assistant expert en analyse de contenu vidéo.

Voici la transcription complète d'une vidéo (avec les timecodes en secondes) :

---
{transcript_with_timecodes}
---

Durée totale : {duration:.0f} secondes

Génère une analyse structurée en JSON avec exactement ce format :
{{
  "title": "Titre accrocheur et descriptif de la vidéo",
  "summary": "Résumé complet en 4-6 phrases couvrant les points essentiels",
  "description": "Description courte (2-3 phrases) style SEO/YouTube",
  "category": "Catégorie principale (ex: Tutoriel, Interview, Conférence, Actualité, Divertissement, Formation, etc.)",
  "keywords": ["mot-clé1", "mot-clé2", "mot-clé3", "mot-clé4", "mot-clé5"],
  "chapters": [
    {{
      "start": 0.0,
      "end": 120.0,
      "title": "Titre du chapitre",
      "description": "Description du chapitre en 1-2 phrases",
      "keywords": ["kw1", "kw2"]
    }}
  ]
}}

Langue de réponse : {language}
Réponds UNIQUEMENT avec le JSON, sans texte avant ou après, sans balises markdown."""


def _build_transcript_with_timecodes(transcription: TranscriptionResult) -> str:
    lines = []
    for seg in transcription.segments:
        lines.append(f"[{seg.start:.1f}s] {seg.text.strip()}")
    return "\n".join(lines)


class Summarizer:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:32b",
        language: str = "fr",
        timeout: int = 300,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.language = language
        self.timeout = timeout

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 2048,
            },
        }

        logger.debug(f"Calling Ollama model '{self.model}' at {url}")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")

    def _parse_json_response(self, raw: str) -> dict:
        # Strip markdown code fences if present
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Trying to extract JSON block...")
            # Try to find JSON object in the response
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")

    def summarize(self, transcription: TranscriptionResult) -> SummaryResult:
        """
        Generate title, summary, description, and chapters from a transcription.
        
        Args:
            transcription: TranscriptionResult from Transcriber
            
        Returns:
            SummaryResult with all generated content
        """
        if not transcription.segments:
            logger.warning("Empty transcription, skipping summarization.")
            return SummaryResult(
                title="[Aucun contenu détecté]",
                summary="",
                description="",
                category="Inconnu",
                keywords=[],
                chapters=[],
                language=self.language,
            )

        transcript_with_timecodes = _build_transcript_with_timecodes(transcription)
        prompt = PROMPT_TEMPLATE.format(
            transcript_with_timecodes=transcript_with_timecodes,
            duration=transcription.duration,
            language=self.language,
        )

        logger.info(f"Sending {len(prompt)} chars to Ollama ({self.model})...")
        raw_response = self._call_ollama(prompt)
        logger.info("LLM response received.")

        data = self._parse_json_response(raw_response)

        chapters = []
        for ch in data.get("chapters", []):
            chapters.append(
                ChapterSummary(
                    start=float(ch.get("start", 0)),
                    end=float(ch.get("end", 0)),
                    title=ch.get("title", ""),
                    description=ch.get("description", ""),
                    keywords=ch.get("keywords", []),
                )
            )

        return SummaryResult(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            keywords=data.get("keywords", []),
            chapters=chapters,
            language=self.language,
        )

    def check_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {models}. "
                    f"Run: ollama pull {self.model}"
                )
            return available
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            return False
