"""Transcription accuracy metrics.

Word error rate is the standard measure in this field and the number the
reference papers quote, so it is what the pipeline is checked against.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["normalise_text", "edit_distance", "word_error_rate", "ErrorRate", "corpus_wer"]

_PUNCTUATION = re.compile(r"[^\w\s']")
_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Reference transcripts and model output differ in casing and punctuation
    conventions; comparing them raw would report errors that are not errors.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ErrorRate:
    errors: int
    total: int

    @property
    def rate(self) -> float:
        return self.errors / self.total if self.total else 0.0

    @property
    def percent(self) -> float:
        return 100.0 * self.rate

    def __str__(self) -> str:
        return f"{self.percent:.1f}% ({self.errors}/{self.total})"


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    reference_words = normalise_text(reference).split()
    hypothesis_words = normalise_text(hypothesis).split()
    return ErrorRate(
        errors=edit_distance(reference_words, hypothesis_words),
        total=len(reference_words),
    )


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    reference_characters = list(normalise_text(reference).replace(" ", ""))
    hypothesis_characters = list(normalise_text(hypothesis).replace(" ", ""))
    return ErrorRate(
        errors=edit_distance(reference_characters, hypothesis_characters),
        total=len(reference_characters),
    )


def corpus_wer(pairs: Sequence[tuple[str, str]]) -> ErrorRate:
    """Aggregate WER over ``(reference, hypothesis)`` pairs.

    Errors and reference lengths are summed before dividing, which is how the
    literature reports corpus WER; averaging per-utterance rates would
    over-weight short utterances.
    """
    errors = 0
    total = 0
    for reference, hypothesis in pairs:
        rate = word_error_rate(reference, hypothesis)
        errors += rate.errors
        total += rate.total
    return ErrorRate(errors=errors, total=total)
