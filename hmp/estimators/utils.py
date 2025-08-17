"""Shared utilities for parameter estimation methods."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ConvergenceChecker(ABC):
    """Abstract base class for convergence checking."""

    @abstractmethod
    def check_convergence(self, current_value: float, previous_values: list) -> bool:
        """Check if convergence has been reached."""
        pass


class RelativeLikelihoodConvergence(ConvergenceChecker):
    """Convergence checker based on relative likelihood improvement.

    Parameters
    ----------
    tolerance : float
        Relative improvement threshold for convergence
    min_iterations : int
        Minimum number of iterations before checking convergence
    """

    def __init__(self, tolerance: float = 1e-4, min_iterations: int = 1):
        self.tolerance = tolerance
        self.min_iterations = min_iterations

    def check_convergence(self, current_value: float, previous_values: list) -> bool:
        """Check convergence based on relative likelihood improvement."""
        if len(previous_values) < self.min_iterations:
            return False

        if len(previous_values) == 0:
            return False

        prev_value = previous_values[-1]

        # Handle edge cases
        if np.isneginf(current_value) or np.isneginf(prev_value):
            return True

        if np.abs(prev_value) < 1e-15:  # Avoid division by very small numbers
            return np.abs(current_value - prev_value) < self.tolerance

        relative_improvement = (current_value - prev_value) / np.abs(prev_value)
        return relative_improvement < self.tolerance


class ParameterConvergence(ConvergenceChecker):
    """Convergence checker based on parameter changes.

    Parameters
    ----------
    tolerance : float
        Parameter change threshold for convergence
    norm : str
        Norm to use for parameter difference ('l2', 'l1', 'inf')
    """

    def __init__(self, tolerance: float = 1e-6, norm: str = 'l2'):
        self.tolerance = tolerance
        self.norm = norm
        self.previous_params = None

    def check_convergence(self, current_params: np.ndarray, previous_values: list = None) -> bool:
        """Check convergence based on parameter changes."""
        if self.previous_params is None:
            self.previous_params = current_params
            return False

        if self.norm == 'l2':
            param_diff = np.linalg.norm(current_params - self.previous_params)
        elif self.norm == 'l1':
            param_diff = np.sum(np.abs(current_params - self.previous_params))
        elif self.norm == 'inf':
            param_diff = np.max(np.abs(current_params - self.previous_params))
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        self.previous_params = current_params
        return param_diff < self.tolerance


def compute_log_likelihood(eventprobs: np.ndarray) -> float:
    """Compute log-likelihood from event probabilities.

    Parameters
    ----------
    eventprobs : np.ndarray
        Event probabilities array of shape (max_duration, n_trials, n_events)

    Returns
    -------
    float
        Log-likelihood value
    """
    # Sum over the first event to get trial likelihoods
    trial_likelihoods = eventprobs[:, :, 0].sum(axis=0)

    # Avoid log(0) by clipping to small positive value
    trial_likelihoods = np.clip(trial_likelihoods, 1e-15, None)

    return np.sum(np.log(trial_likelihoods))


def validate_parameters(channel_pars: np.ndarray, time_pars: np.ndarray,
                       n_events: int, n_dims: int, n_stages: int) -> None:
    """Validate parameter arrays have correct shapes and values.

    Parameters
    ----------
    channel_pars : np.ndarray
        Channel parameters array
    time_pars : np.ndarray
        Time distribution parameters array
    n_events : int
        Expected number of events
    n_dims : int
        Expected number of dimensions/channels
    n_stages : int
        Expected number of stages

    Raises
    ------
    ValueError
        If parameters have incorrect shapes or invalid values
    """
    # Check channel parameters
    if channel_pars.ndim < 2:
        raise ValueError(f"channel_pars must be at least 2D, got {channel_pars.ndim}D")

    expected_channel_shape = (n_events, n_dims)
    if channel_pars.shape[-2:] != expected_channel_shape:
        raise ValueError(f"channel_pars shape {channel_pars.shape} incompatible with "
                        f"expected shape ending in {expected_channel_shape}")

    # Check time parameters
    if time_pars.ndim < 2:
        raise ValueError(f"time_pars must be at least 2D, got {time_pars.ndim}D")

    expected_time_shape = (n_stages, 2)  # shape and scale parameters
    if time_pars.shape[-2:] != expected_time_shape:
        raise ValueError(f"time_pars shape {time_pars.shape} incompatible with "
                        f"expected shape ending in {expected_time_shape}")

    # Check for non-negative time parameters (scale parameters must be positive)
    if np.any(time_pars[..., 1] <= 0):
        raise ValueError("Scale parameters (time_pars[..., 1]) must be positive")

    if np.any(time_pars[..., 0] <= 0):
        raise ValueError("Shape parameters (time_pars[..., 0]) must be positive")


def initialize_parameters(n_events: int, n_dims: int, method: str = 'random',
                         distribution=None, max_scale: float = None,
                         random_seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Initialize channel and time parameters.

    Parameters
    ----------
    n_events : int
        Number of events
    n_dims : int
        Number of dimensions/channels
    method : str
        Initialization method ('random', 'uniform', 'zeros')
    distribution : object, optional
        Distribution object with shape and mean_to_scale methods
    max_scale : float, optional
        Maximum scale for time parameters
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Initialized (channel_pars, time_pars)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize channel parameters
    if method == 'random':
        channel_pars = np.random.randn(n_events, n_dims) * 0.1
    elif method == 'uniform':
        channel_pars = np.ones((n_events, n_dims)) * 0.1
    elif method == 'zeros':
        channel_pars = np.zeros((n_events, n_dims))
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    # Initialize time parameters
    n_stages = n_events + 1
    time_pars = np.zeros((n_stages, 2))

    if distribution is not None and max_scale is not None:
        # Use distribution-specific initialization
        time_pars[:, 0] = distribution.shape  # shape parameters
        stage_durations = np.random.uniform(0, max_scale / n_stages, n_stages)
        time_pars[:, 1] = [distribution.mean_to_scale(d) for d in stage_durations]
    else:
        # Default initialization
        time_pars[:, 0] = 2.0  # Default shape
        time_pars[:, 1] = 1.0  # Default scale

    return channel_pars, time_pars
