"""Model registry.

The single place that knows which adapters exist. Application code asks for a
model by name and receives something implementing
:class:`~ml.models.base.VisualSpeechModel`.
"""

from __future__ import annotations

from collections.abc import Callable

from ml.models.base import VisualSpeechModel

__all__ = ["register_model", "create_model", "available_models", "is_implemented"]

_REGISTRY: dict[str, Callable[..., VisualSpeechModel]] = {}
#: Adapters that are interface stubs; kept visible so the API can report them.
_STUBS: set[str] = set()
_builtins_loaded = False


def register_model(
    name: str,
    factory: Callable[..., VisualSpeechModel],
    *,
    implemented: bool = True,
) -> None:
    _REGISTRY[name] = factory
    if not implemented:
        _STUBS.add(name)


def available_models() -> dict[str, bool]:
    """Registered model names mapped to whether they are actually implemented."""
    _ensure_builtins()
    return {name: name not in _STUBS for name in sorted(_REGISTRY)}


def is_implemented(name: str) -> bool:
    _ensure_builtins()
    return name in _REGISTRY and name not in _STUBS


def create_model(name: str, **kwargs) -> VisualSpeechModel:
    """Construct an adapter by name. Does not load weights."""
    _ensure_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _ensure_builtins() -> None:
    """Import the bundled adapters on first use.

    Deferred rather than done at import time so that importing the registry does
    not drag in every model's dependencies, and so adapters remain free to
    import from :mod:`ml.decoding` without a cycle.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True

    from ml.models.auto_avsr.adapter import AutoAVSRAdapter
    from ml.models.av_hubert import AVHubertAdapter
    from ml.models.custom import CustomVSRAdapter
    from ml.models.vallr import VALLRAdapter

    register_model("auto_avsr", AutoAVSRAdapter)
    register_model("vallr", VALLRAdapter, implemented=False)
    register_model("av_hubert", AVHubertAdapter, implemented=False)
    register_model("custom", CustomVSRAdapter, implemented=False)
