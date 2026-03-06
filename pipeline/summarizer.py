"""
LLM-based summarization, titling and categorization via Ollama (local inference).
Bilingual support: French + Arabic output.
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
class TopicSummary:
    start: float
    end: float
    title_fr: str
    title_ar: str
    description_fr: str
    description_ar: str
    keywords: list = field(default_factory=list)


@dataclass
class SummaryResult:
    title_fr: str
    title_ar: str
    summary_fr: str
    summary_ar: str
    description_fr: str
    description_ar: str
    category: str
    keywords: list
    topics: list
    language: str = "fr+ar"

    # Legacy compat
    @property
    def title(self): return self.title_fr
    @property
    def summary(self): return self.summary_fr
    @property
    def description(self): return self.description_fr
    @property
    def chapters(self): return self.topics

    def to_markdown(self) -> str:
        def fmt(s):
            m = int(s // 60)
            sec = int(s % 60)
            return f"{m:02d}:{sec:02d}"

        lines = [
            f"# {self.title_fr}",
            f"# {self.title_ar}",
            "",
            f"**Catégorie:** {self.category}",
            f"**Mots-clés:** {', '.join(self.keywords)}",
            "",
            "## Description",
            self.description_fr,
            "",
            self.description_ar,
            "",
            "## Résumé / ملخص",
            self.summary_fr,
            "",
            self.summary_ar,
            "",
        ]

        if self.topics:
            lines.append("## Sujets / المواضيع")
            for t in self.topics:
                lines.append(f"### [{fmt(t.start)} -> {fmt(t.end)}] {t.title_fr} | {t.title_ar}")
                lines.append(f"**FR:** {t.description_fr}")
                lines.append(f"**AR:** {t.description_ar}")
                if t.keywords:
                    lines.append(f"*{', '.join(t.keywords)}*")
                lines.append("")

        return "\n".join(lines)


PROMPT_TEMPLATE = """Tu es un assistant expert en analyse de contenu vidéo arabophone et francophone.

Voici la transcription complète d'une vidéo (avec les timecodes en secondes) :

---
{transcript_with_timecodes}
---

Durée totale : {duration:.0f} secondes

Génère une analyse structurée en JSON avec exactement ce format :
{{
  "title_fr": "Titre accrocheur en français",
  "title_ar": "عنوان جذاب بالعربية",
  "summary_fr": "Résumé complet en français en 4-6 phrases couvrant tous les sujets traités",
  "summary_ar": "ملخص كامل بالعربية في 4-6 جمل يغطي جميع المواضيع المعالجة",
  "description_fr": "Description courte (2-3 phrases) style SEO en français",
  "description_ar": "وصف قصير (2-3 جمل) بأسلوب SEO بالعربية",
  "category": "Catégorie principale (ex: Actualité, Interview, Conférence, Reportage, Sport, Culture, etc.)",
  "keywords": ["mot-clé1", "mot-clé2", "كلمة1", "كلمة2", "كلمة3"],
  "topics": [
    {{
      "start": 0.0,
      "end": 120.0,
      "title_fr": "Titre du sujet en français",
      "title_ar": "عنوان الموضوع بالعربية",
      "description_fr": "Description détaillée du sujet en 2-3 phrases en français",
      "description_ar": "وصف تفصيلي للموضوع في 2-3 جمل بالعربية",
      "keywords": ["kw1", "kw2"]
    }}
  ]
}}

Instructions importantes :
- Identifie TOUS les sujets/reportages distincts traités dans la vidéo dans "topics"
- Chaque sujet doit avoir une description détaillée dans les deux langues (français ET arabe)
- Les timecodes doivent correspondre précisément au début et à la fin de chaque sujet
- Réponds UNIQUEMENT avec le JSON, sans texte avant ou après, sans balises markdown"""


def _build_transcript_with_timecodes(transcription):
    lines = []
    for seg in transcription.segments:
        lines.append(f"[{seg.start:.1f}s] {seg.text.strip()}")
    return "\n".join(lines)


class Summarizer:
    def __init__(self, host="http://localhost:11434", model="qwen2.5:32b", language="fr", timeout=300):
        self.host = host.rstrip("/")
        self.model = model
        self.language = language
        self.timeout = timeout

    def _call_ollama(self, prompt):
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 4096},
        }
        logger.debug(f"Calling Ollama model '{self.model}' at {url}")
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    def _parse_json_response(self, raw):
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Trying to extract JSON block...")
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")

    def summarize(self, transcription):
        if not transcription.segments:
            logger.warning("Empty transcription, skipping summarization.")
            return SummaryResult(
                title_fr="[Aucun contenu détecté]", title_ar="[لم يتم اكتشاف محتوى]",
                summary_fr="", summary_ar="", description_fr="", description_ar="",
                category="Inconnu", keywords=[], topics=[],
            )

        transcript_with_timecodes = _build_transcript_with_timecodes(transcription)
        prompt = PROMPT_TEMPLATE.format(
            transcript_with_timecodes=transcript_with_timecodes,
            duration=transcription.duration,
        )

        logger.info(f"Sending {len(prompt)} chars to Ollama ({self.model})...")
        raw_response = self._call_ollama(prompt)
        logger.info("LLM response received.")

        data = self._parse_json_response(raw_response)

        topics = []
        for t in data.get("topics", []):
            topics.append(TopicSummary(
                start=float(t.get("start", 0)),
                end=float(t.get("end", 0)),
                title_fr=t.get("title_fr", t.get("title", "")),
                title_ar=t.get("title_ar", ""),
                description_fr=t.get("description_fr", t.get("description", "")),
                description_ar=t.get("description_ar", ""),
                keywords=t.get("keywords", []),
            ))

        return SummaryResult(
            title_fr=data.get("title_fr", data.get("title", "")),
            title_ar=data.get("title_ar", ""),
            summary_fr=data.get("summary_fr", data.get("summary", "")),
            summary_ar=data.get("summary_ar", ""),
            description_fr=data.get("description_fr", data.get("description", "")),
            description_ar=data.get("description_ar", ""),
            category=data.get("category", ""),
            keywords=data.get("keywords", []),
            topics=topics,
        )

    def check_connection(self):
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
            return available
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            return False
