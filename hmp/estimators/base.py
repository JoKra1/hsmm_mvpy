"""Base classes for parameter estimation in HMP models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EstimationResult:
    """Results from parameter estimation.

    Parameters
    ----------
    channel_pars : np.ndarray
        Estimated channel parameters
    time_pars : np.ndarray
        Estimated time distribution parameters
    likelihood : float
        Final log-likelihood value
    converged : bool
        Whether estimation converged
    n_iterations : int
        Number of iterations performed
    diagnostics : dict
        Estimation-specific diagnostic information
    uncertainty : dict, optional
        Parameter uncertainty measures (for Bayesian methods)
    """

    channel_pars: np.ndarray
    time_pars: np.ndarray
    likelihood: float
    converged: bool
    n_iterations: int
    diagnostics: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate result data after initialization."""
        if self.channel_pars is not None and not isinstance(self.channel_pars, np.ndarray):
            self.channel_pars = np.asarray(self.channel_pars)
        if self.time_pars is not None and not isinstance(self.time_pars, np.ndarray):
            self.time_pars = np.asarray(self.time_pars)


class BaseEstimator(ABC):
    """Abstract base class for parameter estimation methods.

    This class defines the interface that all parameter estimation methods
    must implement to work with HMP models.
    """

    def __init__(self, **kwargs):
        """Initialize the estimator with method-specific parameters."""
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(self, trial_data, initial_channel_pars: np.ndarray,
           initial_time_pars: np.ndarray, fixed_channel_pars: list = None,
           fixed_time_pars: list = None, **kwargs) -> EstimationResult:
        """Estimate model parameters.

        Parameters
        ----------
        trial_data : TrialData
            Trial data to fit the model to
        initial_channel_pars : np.ndarray
            Initial channel parameter values
        initial_time_pars : np.ndarray
            Initial time distribution parameter values
        **kwargs
            Method-specific fitting options

        Returns
        -------
        EstimationResult
            Results of parameter estimation
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self._fitted

    def get_method_name(self) -> str:
        """Get the name of the estimation method."""
        return self.__class__.__name__

    def supports_uncertainty(self) -> bool:
        """Whether this estimator provides uncertainty estimates."""
        return False
