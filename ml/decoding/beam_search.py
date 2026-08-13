"""Joint CTC/attention beam search and confidence scoring.

Wraps the reference ``BatchBeamSearch`` so the rest of the application only sees
:class:`~ml.models.base.Hypothesis` objects with a usable confidence value.
"""

from __future__ import annotations

import math

import torch

from ml.config import DecodingConfig
from ml.decoding.text import TextTransform
from ml.paths import ensure_auto_avsr_importable
from ml.types import Hypothesis

__all__ = ["JointBeamSearchDecoder", "score_to_confidence"]


def score_to_confidence(score: float, token_count: int) -> float:
    """Map a beam score to (0, 1] by length normalisation.

    The score is a sum of log probabilities, so dividing by token count gives a
    mean per-token log probability and exponentiating gives a per-token
    likelihood. This ranks hypotheses correctly and separates confident output
    from guesses, but it is not calibrated: treat it as model confidence, never
    as the probability the transcript is right.
    """
    if token_count <= 0:
        return 0.0
    return float(min(1.0, math.exp(score / token_count)))


class JointBeamSearchDecoder:
    """Beam search over the Transformer decoder with CTC prefix rescoring."""

    def __init__(
        self,
        model: torch.nn.Module,
        text_transform: TextTransform,
        config: DecodingConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        ensure_auto_avsr_importable()
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus

        self.config = config or DecodingConfig()
        self.text_transform = text_transform
        self.device = torch.device(device)

        token_list = text_transform.token_list
        scorers = model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(len(token_list))
        weights = {
            "decoder": 1.0 - self.config.ctc_weight,
            "ctc": self.config.ctc_weight,
            "lm": self.config.lm_weight,
            "length_bonus": self.config.length_penalty,
        }

        self._beam_search = BatchBeamSearch(
            beam_size=self.config.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=model.sos,
            eos=model.eos,
            token_list=token_list,
            pre_beam_score_key=None if self.config.ctc_weight == 1.0 else "decoder",
        )
        self._beam_search.to(device=self.device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def decode(self, encoder_features: torch.Tensor) -> list[Hypothesis]:
        """Decode encoder output ``(T, D)`` into ranked hypotheses."""
        if encoder_features.ndim == 3:
            if encoder_features.shape[0] != 1:
                raise ValueError("Batched decoding is not supported; pass one clip")
            encoder_features = encoder_features.squeeze(0)

        raw_hypotheses = self._beam_search(encoder_features)
        wanted = min(len(raw_hypotheses), max(1, self.config.n_best))

        hypotheses: list[Hypothesis] = []
        for candidate in raw_hypotheses[:wanted]:
            token_ids = self.text_transform.strip_special(candidate.yseq.tolist())
            hypotheses.append(
                Hypothesis(
                    text=self.text_transform.post_process(token_ids),
                    score=float(candidate.score),
                    confidence=score_to_confidence(
                        float(candidate.score), len(token_ids)
                    ),
                    token_ids=token_ids,
                )
            )
        return hypotheses
