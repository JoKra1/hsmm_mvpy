"""Different methods for fitting events."""

from hmp.models.cumulative import CumulativeMethod
from hmp.models.eliminative import EliminativeMethod
from hmp.models.event import EventModel

__all__ = ["EventModel", "EliminativeMethod", "CumulativeMethod"]
