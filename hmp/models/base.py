"""Models to estimate event probabilities."""
from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

import numpy as np

from hmp.distributions import Gamma
from hmp.patterndata import PatternData
from hmp.patterns import HalfSine, Pattern


def fit_likelihoods(event_model) -> np.ndarray:
    """Log-likelihoods of a fitted model, whichever estimator produced it.

    Parameters
    ----------
    event_model : EventModel
        A fitted model.

    Returns
    -------
    np.ndarray
        One value per starting point for EM, one value for a sampler.
    """
    likelihoods = getattr(event_model, "lkhs", None)
    if likelihoods is not None:
        return np.atleast_1d(np.asarray(likelihoods, dtype=float))
    result = getattr(event_model, "estimation_result", None)
    if result is not None:
        return np.atleast_1d(np.asarray(result.likelihood, dtype=float))
    raise AttributeError("fitted model exposes no likelihood")


def max_explored_scale(event_model, n_events: int) -> float:
    """Furthest the fit places the first n_events - 1 stages, used to skip ahead.

    Parameters
    ----------
    event_model : EventModel
        A fitted model.
    n_events : int
        Number of events in the fit.

    Returns
    -------
    float
        Maximum over EM iterations, over posterior draws, or the fitted value.
    """
    deviations = getattr(event_model, "time_pars_dev", None)
    if deviations is not None:
        return float(np.max([np.sum(x[0, : n_events - 1, 1]) for x in deviations]))

    result = getattr(event_model, "estimation_result", None)
    idata = result.diagnostics.get("idata") if result is not None else None
    if idata is not None and "scale" in getattr(idata, "posterior", {}):
        draws = np.asarray(idata.posterior["scale"].values)
        draws = draws.reshape(-1, draws.shape[-1])
        return float(np.max(draws[:, : n_events - 1].sum(axis=1)))

    time_pars = np.asarray(event_model.time_pars)
    return float(np.sum(time_pars[0, : n_events - 1, 1]))


class BaseModel(ABC):
    """The model to analyze the cross-correlated data.

    Parameters
    ----------
    pattern : Pattern
        The pattern and properties to use for cross-correlation. Default is
        half sine with 50 ms width.
    distribution : str
        Probability distribution for the by-trial onset of stages can be
        one of 'gamma','lognormal','wald', or 'weibull'
    """

    def __init__(
        self,
        pattern: Pattern = None,
        distribution: Any = None
    ):
        # default pattern is HalfSine, 50 ms width
        if pattern is None:
            pattern = HalfSine(width=50)
        self.pattern = pattern

        if distribution is None:
            distribution = Gamma()
        self.distribution = distribution
        self._fitted = False

    def _check_fitted(self, op):
        if not self._fitted:
            raise ValueError(f"Cannot {op}, because the model has not been fitted yet.")

    def _check_iterative(self, op, attribute):
        """Refuse a diagnostic that only an iterative estimator produces.

        Takes the name rather than the value: the attribute exists only once a
        fit has stored it, so reading it at the call site raises before the
        unfitted case can be reported.
        """
        self._check_fitted(op)
        if getattr(self, attribute, None) is None:
            raise AttributeError(
                f"Cannot {op}: it is recorded per iteration, and this model was "
                "fitted with an estimator that does not iterate."
            )

    def _instantiate_data_pattern(self, data):
        """Load data pattern, cross-correlate if needed.

        If data is PatternData object, use directly. Otherwise
        create pattern template and do cross correlation.
        If previously fitted (ie transform()), use existing pattern

        """
        if isinstance(data, PatternData):
            pattern_data = data
            if self._fitted and data.pattern != self.pattern:
                warn(f"Cross-correlation pattern {data.pattern}is different in provided data "
                     f"than in model {self.pattern}. Data pattern is used.")
            self.pattern = data.pattern
        else: #assume preprocessed (is checked later)
            pattern_data = PatternData.from_basedata(data, self.pattern)
        return pattern_data

    def _time_to_samples(self, time, sfreq):
        """Calculate samples (int) based on time(s).

        Parameters
        ----------
        time : float | ndarray
            Time or times that need to be converted to samples.
        sfreq : sample frequency of data
        """
        return np.ceil(time * sfreq / 1000).astype(int)

    def _compute_max_events(self, pattern_data : PatternData, location : float):
        """Compute max nr of events that fit in trial.

        Parameters
        ----------
        pattern_data : Pattern
            PatternData object
        location : float
            Location in ms.
        """
        min_dur = np.min(pattern_data.durations.values) - 2 * self.distribution.shift
        location_samples = self._time_to_samples(location, pattern_data.sfreq)
        n = int(np.floor(min_dur / location_samples)) + 1
        print('Max event that can be fit given minimum duration of '
              f'{min_dur} is {n}')
        return n

    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def transform(self):
        ...

    def fit_transform(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)

        cpus = kwargs['cpus'] if 'cpus' in kwargs else 1
        return self.transform(data, cpus=cpus)
