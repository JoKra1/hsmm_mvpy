"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import version, PackageNotFoundError

from . import loocv, models, utils, visu, io, preprocessing, patterns, distributions

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "preprocessing", "patterns",
           "distributions" ,"mcca", "loocv", "__version__"]
