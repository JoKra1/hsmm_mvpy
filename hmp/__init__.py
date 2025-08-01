"""Software for fitting HMP on EEG/MEG data."""

from importlib.metadata import PackageNotFoundError, version

from . import distributions, io, loocv, models, patterns, preprocessing, utils, visu

try:
    __version__ = version("hmp")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["models", "simulations", "utils", "visu", "io", "preprocessing", "patterns",
           "distributions" ,"mcca", "loocv", "__version__"]
