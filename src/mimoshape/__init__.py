"""mimoshape -- phase-domain synthesis of MIMO signals with target CSD and moments."""

from .shaper import (
    MomentTarget,
    EndpointTarget,
    CrestTarget,
    FunctionTarget,
    ScaledFunctionTarget,
    SynthesisProblem,
    MimoShaper,
)
from . import moments
from . import estimate
from . import stationarity
from . import multimodel

__all__ = [
    "MomentTarget",
    "EndpointTarget",
    "CrestTarget",
    "FunctionTarget",
    "ScaledFunctionTarget",
    "SynthesisProblem",
    "MimoShaper",
    "moments",
    "estimate",
    "stationarity",
    "multimodel",
]
