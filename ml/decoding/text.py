"""SentencePiece vocabulary and token-id to text conversion.

The vocabulary is not interchangeable: token ids are only meaningful for the
checkpoint trained against this exact unit list, so it is loaded from the
reference checkout rather than rebuilt.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

from ml.paths import SPM_MODEL_PATH, SPM_UNITS_PATH, require_reference_assets

__all__ = ["TextTransform", "load_text_transform"]

_BLANK = "<blank>"
_EOS = "<eos>"
_UNK = "<unk>"
_SPACE = "<space>"
_SENTENCEPIECE_SPACE = "\u2581"


class TextTransform:
    """Maps between text and the unigram-5000 token ids the model emits."""

    def __init__(
        self,
        spm_model_path: str | Path = SPM_MODEL_PATH,
        units_path: str | Path = SPM_UNITS_PATH,
    ) -> None:
        require_reference_assets()
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(spm_model_path))
        units = Path(units_path).read_text(encoding="utf8").splitlines()
        self._unit_to_id = {
            line.split()[0]: int(line.split()[-1]) for line in units if line.strip()
        }
        # Index 0 is the CTC blank; the final index doubles as sos and eos.
        self.token_list: list[str] = [_BLANK, *self._unit_to_id.keys(), _EOS]
        self.ignore_id = -1

    def __len__(self) -> int:
        return len(self.token_list)

    @property
    def eos_id(self) -> int:
        return len(self.token_list) - 1

    @property
    def blank_id(self) -> int:
        return 0

    def tokenize(self, text: str) -> list[int]:
        """Encode text to token ids. Used by evaluation, not by inference."""
        pieces = self.spm.EncodeAsPieces(text)
        return [self._unit_to_id.get(piece, self._unit_to_id[_UNK]) for piece in pieces]

    def post_process(self, token_ids: Iterable[int]) -> str:
        """Decode token ids to a readable transcript."""
        pieces = [
            self.token_list[int(token_id)]
            for token_id in token_ids
            if int(token_id) != self.ignore_id
        ]
        text = "".join(pieces).replace(_SPACE, " ")
        text = text.replace(_SENTENCEPIECE_SPACE, " ")
        return text.replace(_EOS, "").replace(_BLANK, "").strip()

    def strip_special(self, token_ids: Sequence[int]) -> list[int]:
        """Drop the leading sos and any trailing eos from a beam hypothesis."""
        ids = [int(token_id) for token_id in token_ids]
        if ids and ids[0] == self.eos_id:
            ids = ids[1:]
        while ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        return ids


@lru_cache(maxsize=1)
def load_text_transform() -> TextTransform:
    """Process-wide cached vocabulary; loading it costs real time."""
    return TextTransform()
