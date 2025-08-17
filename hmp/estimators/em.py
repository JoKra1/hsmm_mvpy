"""Expectation-Maximization estimator for HMP models."""

from typing import List, Optional
from warnings import warn

import numpy as np

try:
    from .base import BaseEstimator, EstimationResult
except ImportError:
    # Fallback for direct imports
    from base import BaseEstimator, EstimationResult


class EMEstimator(BaseEstimator):
    """Expectation-Maximization parameter estimator.

    This estimator implements the EM algorithm for parameter estimation
    in HMP models, providing maximum likelihood estimates of channel
    and time distribution parameters.

    Parameters
    ----------
    max_iteration : int, optional
        Maximum number of EM iterations. Default is 1000.
    tolerance : float, optional
        Convergence tolerance for likelihood improvement. Default is 1e-4.
    min_iteration : int, optional
        Minimum number of EM iterations. Default is 1.
    """

    def __init__(self, max_iteration: int = 1000, tolerance: float = 1e-4,
                 min_iteration: int = 1, **kwargs):
        """Initialize the EM estimator.

        Parameters
        ----------
        max_iteration : int, optional
            Maximum number of EM iterations. Default is 1000.
        tolerance : float, optional
            Convergence tolerance for likelihood improvement. Default is 1e-4.
        min_iteration : int, optional
            Minimum number of EM iterations. Default is 1.
        **kwargs
            Additional keyword arguments passed to BaseEstimator.
        """
        super().__init__(**kwargs)
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.min_iteration = min_iteration

    def fit(self, trial_data, initial_channel_pars: np.ndarray,
           initial_time_pars: np.ndarray, model=None,
           channel_map: Optional[np.ndarray] = None,
           time_map: Optional[np.ndarray] = None,
           groups: Optional[np.ndarray] = None,
           fixed_channel_pars: Optional[List[int]] = None,
           fixed_time_pars: Optional[List[int]] = None,
           cpus: int = 1, **kwargs) -> EstimationResult:
        """Fit model parameters using expectation-maximization.

        Parameters
        ----------
        trial_data : TrialData
            The trial data to fit the model to
        initial_channel_pars : np.ndarray
            Initial channel parameter values
        initial_time_pars : np.ndarray
            Initial time distribution parameter values
        model : EventModel
            The model instance (needed for method calls)
        channel_map : np.ndarray, optional
            2D array mapping channel parameters to groups
        time_map : np.ndarray, optional
            2D array mapping time parameters to groups
        groups : np.ndarray, optional
            Array indicating the groups for grouping modeling
        fixed_channel_pars : list[int], optional
            Indices of channel parameters to fix during estimation
        fixed_time_pars : list[int], optional
            Indices of time parameters to fix during estimation
        cpus : int, optional
            Number of cores to use. Default is 1.

        Returns
        -------
        EstimationResult
            Results of the EM estimation
        """
        if model is None:
            raise ValueError("EMEstimator requires a model instance to access helper methods")

        assert channel_map.shape[0] == time_map.shape[0], (
            "Both maps need to indicate the same number of groups."
        )

        # Handle multiple starting points
        if initial_channel_pars.ndim == 4:
            # Multiple starting points provided
            n_starting_points = initial_channel_pars.shape[0]
            channel_pars_list = [initial_channel_pars[i] for i in range(n_starting_points)]
            time_pars_list = [initial_time_pars[i] for i in range(n_starting_points)]
        else:
            # Single starting point
            channel_pars_list = [initial_channel_pars]
            time_pars_list = [initial_time_pars]

        # Run EM for each starting point
        results = []
        for c_pars, t_pars in zip(channel_pars_list, time_pars_list):
            result = self._fit_single_starting_point(
                trial_data, c_pars, t_pars, model, fixed_channel_pars,
                fixed_time_pars, channel_map, time_map, groups, cpus
            )
            results.append(result)

        # Select best result based on likelihood
        likelihoods = [r.likelihood for r in results]
        best_idx = np.argmax(likelihoods)

        return results[best_idx]

    def _fit_single_starting_point(self, trial_data, initial_channel_pars: np.ndarray,
                                  initial_time_pars: np.ndarray, model,
                                  fixed_channel_pars: list = None, fixed_time_pars: list = None,
                                  channel_map: np.ndarray = None, time_map: np.ndarray = None,
                                  groups: np.ndarray = None,
                                  cpus: int = 1) -> EstimationResult:
        """Fit parameters for a single starting point."""
        lkh, eventprobs = model._distribute_groups(
            trial_data, initial_channel_pars, initial_time_pars,
            channel_map, time_map, groups, cpus=cpus
        )
        data_groups = np.unique(groups)
        channel_pars = initial_channel_pars.copy()
        time_pars = initial_time_pars.copy()
        traces = [lkh]
        time_pars_dev = [time_pars.copy()]
        i = 0
        lkh_prev = None

        while i < self.max_iteration:  # Expectation-Maximization algorithm
            if i >= self.min_iteration and lkh_prev is not None:
                if (np.isneginf(lkh.sum()) or
                    self.tolerance > (lkh.sum() - lkh_prev.sum()) / np.abs(lkh_prev.sum())):
                    break

            # As long as new run gives better likelihood, go on
            lkh_prev = lkh.copy()

            for cur_group in data_groups:  # get params/c_pars
                channel_map_group = np.where(channel_map[cur_group, :] >= 0)[0]
                time_map_group = np.where(time_map[cur_group, :] >= 0)[0]
                epochs_group = np.where(groups == cur_group)[0]
                # get c_pars/t_pars by group
                channel_pars[cur_group, channel_map_group, :], time_pars[cur_group, time_map_group, :] = (
                    self._get_channel_time_parameters_expectation(
                        trial_data,
                        eventprobs.values[:, :np.max(trial_data.durations[epochs_group]), channel_map_group],
                        model,
                        subset_epochs=epochs_group,
                    )
                )

                if fixed_channel_pars is not None and len(fixed_channel_pars) > 0:
                    channel_pars[cur_group, fixed_channel_pars, :] = initial_channel_pars[
                        cur_group, fixed_channel_pars, :
                    ].copy()
                if fixed_time_pars is not None and len(fixed_time_pars) > 0:
                    time_pars[cur_group, fixed_time_pars, :] = initial_time_pars[
                        cur_group, fixed_time_pars, :
                    ].copy()

            # set c_pars to mean if requested in map
            for m in range(model.n_events):
                for m_set in np.unique(channel_map[:, m]):
                    if m_set >= 0:
                        channel_pars[channel_map[:, m] == m_set, m, :] = np.mean(
                            channel_pars[channel_map[:, m] == m_set, m, :], axis=0
                        )

            # set param to mean if requested in map
            for p in range(model.n_events + 1):
                for p_set in np.unique(time_map[:, p]):
                    if p_set >= 0:
                        time_pars[time_map[:, p] == p_set, p, :] = np.mean(
                            time_pars[time_map[:, p] == p_set, p, :], axis=0
                        )


            lkh, eventprobs = model._distribute_groups(
                trial_data, channel_pars, time_pars, channel_map, time_map, groups, cpus=cpus
            )
            traces.append(lkh)
            time_pars_dev.append(time_pars.copy())
            i += 1

        if i == self.max_iteration:
            warn(
                f"Convergence failed, estimation hit the maximum number of iterations: "
                f"({int(self.max_iteration)})",
                RuntimeWarning,
            )

        # Determine convergence
        converged = len(traces) < self.max_iteration
        n_iterations = len(traces)

        # Create diagnostics
        diagnostics = {
            'traces': np.array(traces),
            'time_pars_dev': np.array(time_pars_dev),
            'method': 'EM',
            'tolerance_achieved': np.abs(traces[-1] - traces[-2]) / np.abs(traces[-2]) if len(traces) > 1 else np.inf
        }

        self._fitted = True

        return EstimationResult(
            channel_pars=channel_pars,
            time_pars=time_pars,
            likelihood=lkh.sum() if hasattr(lkh, 'sum') else lkh,
            converged=converged,
            n_iterations=n_iterations,
            diagnostics=diagnostics
        )

    def _get_channel_time_parameters_expectation(
        self,
        trial_data,
        eventprobs: np.ndarray,
        model,
        subset_epochs: list[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the channel and time parameters using the expectation step."""
        channel_pars = np.zeros((eventprobs.shape[2], model.n_dims))
        # Channel contribution from Expectation, Eq 11 from 2024 paper
        for event in range(eventprobs.shape[2]):
            for comp in range(model.n_dims):
                event_data = np.zeros((len(subset_epochs), np.max(trial_data.durations[subset_epochs])))
                for trial_idx, trial in enumerate(subset_epochs):
                    start, end = trial_data.starts[trial], trial_data.ends[trial]
                    duration = end - start + 1
                    event_data[trial_idx, :duration] = trial_data.cross_corr[start : end + 1, comp]
                channel_pars[event, comp] = np.mean(
                    np.sum(eventprobs[subset_epochs, :, event] * event_data, axis=1)
                )
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trial

        # Time parameters from Expectation Eq 10 from 2024 paper
        # calc averagepos here as mean_d can be group dependent, whereas scale_parameters() assumes
        # it's general
        event_times_mean = np.concatenate(
            [
                np.arange(np.max(trial_data.durations[subset_epochs])) @ eventprobs[subset_epochs].mean(axis=0),
                [np.mean(trial_data.durations[subset_epochs]) - 1],
            ]
        )
        time_pars = self._scale_parameters(model, averagepos=event_times_mean)
        return channel_pars, time_pars

    def _scale_parameters(self, model, averagepos: np.ndarray) -> np.ndarray:
        """Scale parameters from the average position of events."""
        params = np.zeros((len(averagepos), 2), dtype=np.float64)
        params[:, 0] = model.distribution.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = model.distribution.mean_to_scale(params[:, 1])
        return params

    def supports_uncertainty(self) -> bool:
        """EM provides point estimates, not uncertainty."""
        return False
