"""PII detection + redaction via Microsoft Presidio — the OSS standard (Analyzer + Anonymizer).

This is the *enforcement* the old regex input ring lacked: we don't just flag PII, we REDACT it
from the query before it's embedded into the vector index or logged to LangSmith. We redact only
unambiguous identifiers (email/phone/SSN/card/…), never PERSON/LOCATION/DATE — those are often
legitimate query content and redacting them would degrade retrieval (see config rationale).

Lazy-loaded (Presidio pulls a spaCy NER pipeline), so importing this module is free.
"""
from __future__ import annotations

from hcft_agent.config import settings


class PIIGuard:
    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None

    def _load(self):
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_anonymizer import AnonymizerEngine

            print("[guard] loading Presidio (PII analyzer + anonymizer) ...")
            self._analyzer = AnalyzerEngine()
            # Presidio's default US_SSN/phone recognizers under-fire (SSN misses canonical
            # formats; US phone scores ~0.4). Register deterministic high-score patterns for the
            # identifiers we must never leak — we own which entities + patterns, Presidio runs them.
            for entity, name, regex in [
                ("US_SSN", "ssn-dashed", r"\b\d{3}-\d{2}-\d{4}\b"),
                ("PHONE_NUMBER", "phone-us", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
                ("CREDIT_CARD", "cc-16", r"\b(?:\d[ -]*?){13,16}\b"),
            ]:
                self._analyzer.registry.add_recognizer(
                    PatternRecognizer(supported_entity=entity,
                                      patterns=[Pattern(name=name, regex=regex, score=0.85)])
                )
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer

    def _analyze(self, text: str):
        analyzer, _ = self._load()
        return analyzer.analyze(
            text=text, language="en",
            entities=list(settings.pii_redact_entities),
            score_threshold=settings.pii_score_threshold,
        )

    def scan(self, text: str) -> list[str]:
        """Return the sorted set of PII entity types found (e.g. ['EMAIL_ADDRESS'])."""
        return sorted({r.entity_type for r in self._analyze(text)})

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Return (redacted_text, entity_types). Replaces each hit with <ENTITY_TYPE>.
        No hits -> original text unchanged."""
        results = self._analyze(text)
        if not results:
            return text, []
        _, anonymizer = self._load()
        red = anonymizer.anonymize(text=text, analyzer_results=results)
        return red.text, sorted({r.entity_type for r in results})
