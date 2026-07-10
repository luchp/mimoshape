"""mimoshape -- phase-domain synthesis of MIMO signals with target CSD and moments."""

from .shaper import MomentTarget, EndpointTarget, SynthesisProblem, MimoShaper
from . import moments
from . import estimate

__all__ = [
    "MomentTarget",
    "EndpointTarget",
    "SynthesisProblem",
    "MimoShaper",
    "moments",
    "estimate",
]
