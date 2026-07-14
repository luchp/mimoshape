"""mimoshape -- phase-domain synthesis of MIMO signals with target CSD and moments."""

from .shaper import MomentTarget, EndpointTarget, CrestTarget, SynthesisProblem, MimoShaper
from . import moments
from . import estimate
from . import stationarity
from . import multimodel

__all__ = [
    "MomentTarget",
    "EndpointTarget",
    "CrestTarget",
    "SynthesisProblem",
    "MimoShaper",
    "moments",
    "estimate",
    "stationarity",
    "multimodel",
]
