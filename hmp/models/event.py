"""Definition of ``EventModel`` class.

The ``EventModel`` class is the base model for estimating
hidden multivariate pattern models.
"""


import itertools
import multiprocessing as mp
from itertools import product

import numpy as np
import xarray as xr
from pandas import MultiIndex

from hmp.models.base import BaseModel
from hmp.trialdata import TrialData

try:
    from hmp.estimators.base import EstimationResult
    from hmp.estimators.em import EMEstimator
except ImportError:
    # Handle case where estimators module is not available
    EstimationResult = None
    EMEstimator = None

try:
    __IPYTHON__
except NameError:
    pass



class EventModel(BaseModel):
    """
    A model for estimating HMP events.

    Parameters
    ----------
    n_events : int
        The number of HMP events to estimate.
    fixed_time_pars : list, optional
        List of time parameters to fix during estimation.
        If None, all time parameters are estimated.
    fixed_channel_pars : list, optional
        List of channel parameters to fix during estimation.
        If None, all channel parameters are estimated.
    tolerance : float, optional
        Convergence tolerance for the expectation maximization algorithm. Default is 1e-4.
    max_iteration : int, optional
        Maximum number of iterations for the expectation maximization algorithm. Default is 1e3.
    min_iteration : int, optional
        Minimum number of iterations for the expectation maximization algorithm. Default is 1.
    starting_points : int, optional
        Number of random starting points to use for initialization. Default is 1.
    max_scale : float, optional
        Maximum mean distance between events, used when generating random starting points.
        Default is None.
    """

    def __init__(
        self, *args, n_events: int, fixed_time_pars: list = None, fixed_channel_pars: list = None,
        tolerance: float = 1e-4,
        max_iteration: int = 1e3,
        min_iteration: int = 1,
        starting_points: int = 1,
        max_scale: float = None,
        **kwargs
    ):
        assert np.issubdtype(type(n_events), np.integer), \
         (
             f"An integer for the number of expected transition events"
             f" is expected, got {type(n_events).__name__} instead"
         )

        self.n_events = n_events
        self.n_dims = None
        self.fixed_time_pars = fixed_time_pars
        self.fixed_channel_pars = fixed_channel_pars
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.min_iteration = min_iteration
        self.starting_points = starting_points
        self.max_scale = max_scale
        self.grouping_dict = {}
        self.time_map = np.zeros((1, self.n_events + 1))
        self.channel_map = np.zeros((1, self.n_events))
        super().__init__(*args, **kwargs)

    def _prepare_parameters(self, trial_data: TrialData, channel_pars: np.ndarray = None,
                           time_pars: np.ndarray = None, fixed_time_pars: list = None,
                           fixed_channel_pars: list = None, grouping_dict: dict = None,
                           channel_map: np.ndarray = None, time_map: np.ndarray = None,
                           verbose: bool = True):
        """
        Prepare parameters for estimation.

        Returns
        -------
        tuple
            (n_groups, groups, glabels, channel_map, time_map, channel_pars, time_pars, fixed_channel_pars, fixed_time_pars, grouping_dict)
        """
        if grouping_dict is None:
            grouping_dict = self.grouping_dict
            channel_map = self.channel_map
            time_map = self.time_map
        n_groups, groups, glabels = self.group_constructor(
            trial_data, grouping_dict, channel_map, time_map, verbose
        )

        # Handle fixed parameters
        if self.fixed_time_pars is None:
            fixed_time_pars = [] if fixed_time_pars is None else fixed_time_pars
        else:
            fixed_time_pars = self.fixed_time_pars

        if self.fixed_channel_pars is None:
            fixed_channel_pars = [] if fixed_channel_pars is None else fixed_channel_pars
        else:
            fixed_channel_pars = self.fixed_channel_pars

        if time_pars is None:
            # If no time parameters starting points are provided generate standard ones
            time_pars = (
                np.zeros((n_groups, self.n_events + 1, 2)) * np.nan
            )  # by default nan for missing stages
            for cur_group in range(n_groups):
                time_group = np.where(time_map[cur_group, :] >= 0)[0]
                n_stage_group = len(time_group)
                # by default starting point is to split the average duration in equal bins
                time_pars[cur_group, time_group, :] = np.tile(
                    [
                        self.distribution.shape,
                        self.distribution.mean_to_scale(
                            np.mean(trial_data.durations[groups == cur_group]) / (n_stage_group)
                        ),
                    ],
                    (n_stage_group, 1),
                )

        if channel_pars is None:
            # By defaults c_pars are initiated to 0
            channel_pars = np.zeros((n_groups, self.n_events, self.n_dims), dtype=np.float64)
            if (channel_map < 0).any():  # set missing c_pars to nan
                for cur_group in range(n_groups):
                    channel_pars[cur_group, np.where(channel_map[cur_group, :] < 0)[0], :] = np.nan

        return (n_groups, groups, glabels, channel_map, time_map, channel_pars, time_pars,
                fixed_channel_pars, fixed_time_pars, grouping_dict)

    def fit(self, trial_data: TrialData, initial_channel_pars=None, initial_time_pars=None, estimator=None, **kwargs):
        """
        Fit the model to trial data using the new estimation framework.

        Parameters
        ----------
        trial_data : TrialData
            The trial data to fit the model to.
        initial_channel_pars : np.ndarray, optional
            Initial channel parameters for fitting. If None, will be automatically generated.
        initial_time_pars : np.ndarray, optional
            Initial time parameters for fitting. If None, will be automatically generated.
        estimator : Estimator instance, optional
            An estimator instance (e.g., EMEstimator) configured with the desired parameters.
            If None, will create a default EMEstimator instance.
        **kwargs
            Additional keyword arguments passed to the estimator or used for parameter preparation.

        Returns
        -------
        EstimationResult
            The estimation result containing fitted parameters and metadata.
        """
        # Create default estimator if none provided
        if estimator is None:
            if EMEstimator is None:
                raise ImportError("EMEstimator not available. Please check your installation.")
            estimator = EMEstimator(
                tolerance=kwargs.get('tolerance', self.tolerance),
                max_iteration=kwargs.get('max_iteration', int(self.max_iteration)),
                min_iteration=kwargs.get('min_iteration', self.min_iteration)
            )

        # Prepare parameters using the existing _prepare_parameters method
        # First set n_dims from trial data if not already set
        if self.n_dims is None:
            self.n_dims = trial_data.n_dims

        (n_groups, groups, glabels, channel_map, time_map,
         channel_pars, time_pars, fixed_channel_pars, fixed_time_pars,
         grouping_dict) = self._prepare_parameters(
            trial_data,
            channel_pars=initial_channel_pars,
            time_pars=initial_time_pars,
            fixed_time_pars=kwargs.get('fixed_time_pars'),
            fixed_channel_pars=kwargs.get('fixed_channel_pars'),
            grouping_dict=kwargs.get('grouping_dict'),
            channel_map=kwargs.get('channel_map'),
            time_map=kwargs.get('time_map'),
            verbose=kwargs.get('verbose', True)
        )

        # Fit using the provided estimator
        result = estimator.fit(
            trial_data,
            initial_channel_pars=channel_pars,
            initial_time_pars=time_pars,
            model=self,
            channel_map=channel_map,
            time_map=time_map,
            groups=groups,
            fixed_channel_pars=fixed_channel_pars,
            fixed_time_pars=fixed_time_pars,
            cpus=kwargs.get('cpus', 1),
            **{k: v for k, v in kwargs.items() if k not in
               ['tolerance', 'max_iteration', 'min_iteration', 'fixed_time_pars',
                'fixed_channel_pars', 'grouping_dict', 'channel_map', 'time_map', 'cpus']}
        )

        # Store results in model
        self.channel_pars = result.channel_pars
        self.time_pars = result.time_pars
        self.lkhs = result.likelihood
        self.traces = result.diagnostics.get('traces', np.array([result.likelihood]))
        self.time_pars_dev = result.diagnostics.get('time_pars_dev', result.time_pars[np.newaxis, ...])
        self._fitted = True

        return result

    def transform(self, trial_data: TrialData) -> tuple[np.ndarray, xr.DataArray]:
        """
        Transform the trial data using the fitted model.

        Parameters
        ----------
        trial_data : TrialData
            The trial data to transform.

        Returns
        -------
        likelihoods : list
            List of log-likelihoods for each submodel (number of events).
        xr_eventprobs : xr.DataArray
            Concatenated event probability arrays for all submodels, indexed by number of events.
        """
        _, groups, glabels = self.group_constructor(
                trial_data, self.grouping_dict
            )
        likelihoods, xreventprobs = self._distribute_groups(
            trial_data, self.channel_pars, self.time_pars,
            self.channel_map, self.time_map, groups, False
        )
        return likelihoods, xreventprobs



    @property
    def xrtraces(self):
        """
        Returns the traces of the log-likelihood for each EM iteration as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("em_iteration", "group") containing
            the log-likelihood traces.
        """
        self._check_fitted("get traces")
        return xr.DataArray(
            self.traces,
            dims=("em_iteration", "group"),
            name="traces",
            coords={
                "em_iteration": range(self.traces.shape[0]),
                "group": range(self.traces.shape[1]),
            }
        )

    @property
    def xrlikelihoods(self):
        """
        Returns the log-likelihoods as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the log-likelihood values.
        """
        self._check_fitted("get likelihoods")
        return xr.DataArray(self.lkhs, name="loglikelihood")

    @property
    def xrtime_pars_dev(self):
        """
        Returns the time parameter for each EM iteration as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("em_iteration", "group", "stage", "time_pars")
            containing the time parameter deviations.
        """
        self._check_fitted("get dev time pars")
        return xr.DataArray(
            self.time_pars_dev,
            dims=("em_iteration", "group", "stage", "time_pars"),
            name="time_pars_dev",
            coords=[
                range(self.time_pars_dev.shape[0]),
                range(self.time_pars_dev.shape[1]),
                range(self.n_events + 1),
                ["shape", "scale"],
            ],
        )

    @property
    def xrtime_pars(self):
        """
        Returns the time parameters as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("group", "stage", "parameter") containing the time
            parameters.
        """
        self._check_fitted("get xrtime_pars")
        return xr.DataArray(
            self.time_pars,
            dims=("group", "stage", "parameter"),
            name="time_pars",
            coords={
                "group": range(self.time_pars.shape[0]),
                "stage": range(self.n_events + 1),
                "parameter": ["shape", "scale"],
            },
        )

    @property
    def xrchannel_pars(self):
        """
        Returns the channel parameters as an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions ("group", "event", "channel") containing the channel
            parameters.
        """
        self._check_fitted("get xrchannel_pars")
        return xr.DataArray(
            self.channel_pars,
            dims=("group", "event", "channel"),
            name="channel_pars",
            coords={
                "group": range(self.channel_pars.shape[0]),
                "event": range(self.n_events),
                "channel": range(self.n_dims),
            },
        )

    def get_channel_time_parameters_expectation(
        self,
        trial_data: TrialData,
        eventprobs: np.ndarray,
        subset_epochs: list[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the channel and time parameters using the expectation step.

        Parameters
        ----------
        trial_data : TrialData
            The trial data containing cross-correlation and event information.
        eventprobs : np.ndarray
            A 3D array of shape (n_trials, max_duration, n_events) containing the event
            probabilities.
        subset_epochs : list[int], optional
            A list of trial indices to consider for the computation. If None, all trials are used.

        Returns
        -------
        channel_pars : np.ndarray
            A 2D array of shape (n_events, n_dims) with the estimated channel parameters.
        time_pars : np.ndarray
            A 2D array of shape (n_stages, 2) with the estimated time parameters (shape and scale).
        """
        channel_pars = np.zeros((eventprobs.shape[2], self.n_dims))
        # Channel contribution from Expectation, Eq 11 from 2024 paper
        for event in range(eventprobs.shape[2]):
            for comp in range(self.n_dims):
                event_data = np.zeros((len(subset_epochs),
                                       np.max(trial_data.durations[subset_epochs])))
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
                np.arange(np.max(trial_data.durations[subset_epochs])) @ eventprobs[
                    subset_epochs].mean(axis=0),
                [np.mean(trial_data.durations[subset_epochs]) - 1],
            ]
        )
        time_pars = self.scale_parameters(averagepos=event_times_mean)
        return channel_pars, time_pars

    def gen_random_stages(self, n_events: int) -> np.ndarray:
        """
        Compute random stage durations.

        Generates random stage durations between 0 and the mean reaction time (RT) by iteratively
        drawing samples from a uniform distribution. The last stage duration is computed as
        1 minus the cumulative duration of previous stages.
        The stages are then scaled to the mean RT.

        Parameters
        ----------
        n_events : int
            The number of events to generate random durations for.

        Returns
        -------
        np.ndarray
            A 2D array where each row contains the shape and scale parameters for a stage.
        """
        rnd_durations = np.zeros(n_events + 1)
        assert self.event_width_samples*(n_events + 1) < self.max_scale, \
            f"Max_scale too short, need to be more than {self.event_width_samples*(n_events+1)}"
        while any(rnd_durations < self.event_width_samples):  # at least event_width
            rnd_events = np.random.default_rng().integers(
                low=0, high=self.max_scale, size=n_events
            )  # n_events between 0 and mean_d
            rnd_events = np.sort(rnd_events)
            rnd_durations = np.hstack((rnd_events, self.max_scale)) - np.hstack(
                (0, rnd_events)
            )  # associated durations
        random_stages = np.array(
            [[self.distribution.shape, self.distribution.mean_to_scale(x)] for x in rnd_durations]
        )
        return random_stages


    def scale_parameters(self, averagepos: np.ndarray) -> np.ndarray:
        """
        Scale parameters from the average position of events.

        This method is used during the re-estimation step in the EM procedure.
        It computes the likeliest location of events from `eventprobs` and calculates
        the scale parameters as the average distance between consecutive events.

        Parameters
        ----------
        averagepos : np.ndarray
            A 1D array containing the average positions of events.

        Returns
        -------
        np.ndarray
            A 2D array where each row contains the shape and scale parameters
            for the corresponding event distribution.
        """
        params = np.zeros((len(averagepos), 2), dtype=np.float64)
        params[:, 0] = self.distribution.shape
        params[:, 1] = np.diff(averagepos, prepend=0)
        params[:, 1] = self.distribution.mean_to_scale(params[:, 1])
        return params

    def estim_probs(
        self,
        trial_data: TrialData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        location: bool = True,
        subset_epochs: list[int] | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Estimate probabilities for events and compute the log-likelihood.

        Parameters
        ----------
        trial_data : TrialData
            The trial data containing cross-correlation and event information.
        channel_pars : np.ndarray
            A 2D array of shape (n_events, n_channels) or a 3D array of shape
            (iteration, n_events, n_channels) containing initial conditions for
            channel contributions to events.
        time_pars : np.ndarray
            A 2D array of shape (n_stages, n_parameters) or a 3D array of shape
            (iteration, n_stages, n_parameters) containing initial conditions for
            the distribution parameters.
        location : bool, optional
            Whether to add a minimum distance between events to avoid event collapse
            during the expectation-maximization algorithm. Default is True.
        subset_epochs : list[int] or None, optional
            A list of trial indices to consider for the computation. If None, all trials
            are used. Default is None.

        Returns
        -------
        loglikelihood : float
            The summed log probabilities.
        eventprobs : np.ndarray
            A 3D array of shape (n_trials, max_samples, n_events) containing the probabilities
            for each event.
        """
        n_events = channel_pars.shape[0]
        n_stages = n_events + 1
        locations = np.zeros(n_stages, dtype=int)
        if location:
            locations[1:-1] = self.location
        if subset_epochs is not None:
            if len(subset_epochs) == trial_data.n_trials:  # boolean indices
                subset_epochs = np.where(subset_epochs)[0]
        n_trials = len(subset_epochs)
        durations = trial_data.durations[subset_epochs]
        starts = trial_data.starts[subset_epochs]
        ends = trial_data.ends[subset_epochs]
        max_duration = np.max(durations)
        gains = np.zeros((trial_data.n_samples, n_events), dtype=np.float64)
        for i in range(trial_data.n_dims):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = (
                gains
                + trial_data.cross_corr[:, i][np.newaxis].T * channel_pars[:, i]
                - channel_pars[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros([max_duration, n_trials, n_events], dtype=np.float64)  # prob per trial
        probs_b = np.zeros(
            [max_duration, n_trials, n_events], dtype=np.float64
        )  # Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :]
            # Same but sample and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = gains[starts[trial] : ends[trial] + 1, :][
                ::-1, ::-1
            ]

        pmf = np.zeros([max_duration, n_stages], dtype=np.float64)  # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(1e-15, locations[stage]),
                    self.distribution_pdf(time_pars[stage, 0], time_pars[stage, 1], max_duration)[
                        locations[stage] :
                    ],
                )
            )
        pmf_b = pmf[:, ::-1]  # Stage reversed gamma pmf, same order as prob_b

        forward = np.zeros((max_duration, n_trials, n_events), dtype=np.float64)
        backward = np.zeros((max_duration, n_trials, n_events), dtype=np.float64)
        # Computing forward and backward helper variable
        #  when stage = 0:
        forward[:, :, 0] = (
            np.tile(pmf[:, 0][np.newaxis].T, (1, n_trials)) * probs[:, :, 0]
        )  # first stage transition is p(B) * p(d)
        backward[:, :, 0] = np.tile(
            pmf_b[:, 0][np.newaxis].T, (1, n_trials)
        )  # Reversed gamma (i.e. last stage) without probs as last event ends at time T

        for event in np.arange(
            1, n_events
        ):  # Following stage transitions integrate previous transitions
            add_b = backward[:, :, event - 1] * probs_b[:, :, event - 1]  # Next stage in back
            for trial in np.arange(n_trials):
                # convolution between gamma * gains at previous event and event
                forward[:, trial, event] = np.convolve(forward[:, trial, event - 1], pmf[:, event])[
                    : max_duration
                ]
                # same but backwards
                backward[:, trial, event] = np.convolve(add_b[:, trial], pmf_b[:, event])[
                    : max_duration
                ]
            forward[:, :, event] = forward[:, :, event] * probs[:, :, event]
        # re-arranging backward to the expected variable
        backward = backward[:, :, ::-1]  # undoes stage inversion
        for trial in np.arange(n_trials):  # Undoes sample inversion
            backward[: durations[trial], trial, :] = backward[: durations[trial], trial, :][::-1]
        eventprobs = forward * backward
        eventprobs = np.clip(eventprobs, 0, None)  # floating point precision error
        likelihood = np.sum(
            np.log(eventprobs[:, :, 0].sum(axis=0))
        )  # sum over max_samples to avoid 0s in log
        eventprobs = eventprobs / eventprobs.sum(axis=0)
        eventprobs = eventprobs.transpose((1,0,2))
        return [likelihood, eventprobs]

    def _distribute_groups(
        self,
        trial_data: TrialData,
        channel_pars: np.ndarray,
        time_pars: np.ndarray,
        channel_map: np.ndarray,
        time_map: np.ndarray,
        groups: np.ndarray,
        location: bool = True,
        cpus: int = 1,
    ) -> tuple[np.ndarray, xr.DataArray]:
        """
        Estimate probability groups for grouping models.

        This method computes the log-likelihood and event probabilities for each group
        in the grouping model, using the provided channel and time parameters.

        Parameters
        ----------
        trial_data : TrialData
            The trial data containing cross-correlation and event information.
        channel_pars : np.ndarray
            A 2D array of shape (n_events, n_channels) or a 3D array of shape
            (iteration, n_events, n_channels) containing initial conditions for
            channel contributions to events.
        time_pars : np.ndarray
            A 2D array of shape (n_stages, n_parameters) or a 3D array of shape
            (iteration, n_stages, n_parameters) containing initial conditions for
            the distribution parameters.
        channel_map : np.ndarray
            A 2D array mapping channel parameters to groups.
        time_map : np.ndarray
            A 2D array mapping time parameters to groups.
        groups : np.ndarray
            An array indicating the groups for grouping modeling.
        location : bool, optional
            Whether to add a minimum distance between events to avoid event collapse
            during the expectation-maximization algorithm. Default is True.
        cpus : int, optional
            Number of cores to use in multiprocessing functions. Default is 1.

        Returns
        -------
        loglikelihood : np.ndarray
            A 1D array of log-likelihood values for each group.
        all_xreventprobs : xr.DataArray
            An xarray DataArray containing event probabilities with dimensions
            ("trial", "sample", "event").
        """
        data_groups = np.unique(groups)
        likes_events_group = []
        all_xreventprobs = []
        if cpus > 1:
            with mp.Pool(processes=cpus) as pool:
                likes_events_group = pool.starmap(
                    self.estim_probs,
                    zip(
                        itertools.repeat(trial_data),
                        [channel_pars[cur_group, channel_map[cur_group, :] >= 0, :]
                         for cur_group in data_groups],
                        [time_pars[cur_group, time_map[cur_group, :] >= 0, :]
                         for cur_group in data_groups],
                        itertools.repeat(location),
                        [groups == cur_group for cur_group in data_groups],
                        itertools.repeat(False),
                    ),
                )
        else:
            for cur_group in data_groups:
                channel_pars_group = channel_pars[
                    cur_group, channel_map[cur_group, :] >= 0, :
                ]  # select existing magnitudes
                # select existing params
                time_pars_group = time_pars[cur_group, time_map[cur_group, :] >= 0, :]
                likes_events_group.append(
                    self.estim_probs(
                        trial_data,
                        channel_pars_group,
                        time_pars_group,
                        location,
                        subset_epochs=(groups == cur_group),
                    )
                )

        likelihood = np.array([x[0] for x in likes_events_group])

        for i, cur_group in enumerate(data_groups):
            part = trial_data.xrdurations.coords["participant"].values[(groups == cur_group)]
            epoch = trial_data.xrdurations.coords["epoch"].values[(groups == cur_group)]
            data_events =  channel_map[cur_group, :] >= 0
            trial_x_part = xr.Coordinates.from_pandas_multiindex(
                MultiIndex.from_arrays([part, epoch], names=("participant", "epoch")),
                "trial",
            )
            xreventprobs = xr.DataArray(likes_events_group[i][1], dims=("trial", "sample", "event"),
                coords={
                    "event": ("event", np.arange(self.n_events)[data_events]),
                    "sample": ("sample", range(np.shape(likes_events_group[i][1])[1])),
                },
            )
            xreventprobs = xreventprobs.assign_coords(trial_x_part)
            xreventprobs = xreventprobs.assign_coords(group=("trial", groups[groups == cur_group],))
            all_xreventprobs.append(xreventprobs)
        all_xreventprobs = xr.concat(all_xreventprobs, dim="trial")
        all_xreventprobs.attrs['sfreq'] = self.sfreq
        all_xreventprobs.attrs['event_width_samples'] = self.event_width_samples
        return [np.array(likelihood), all_xreventprobs]

    def distribution_pdf(self, shape: float, scale: float, max_duration: int) -> np.ndarray:
        """
        Return a discretized probability density function (PDF) for a provided scipy distribution.

        This method computes the PDF using the given shape and scale parameters over a range
        from 0 to `max_duration`, and normalizes it to ensure the probabilities sum to 1.

        Parameters
        ----------
        shape : float
            The shape parameter of the distribution.
        scale : float
            The scale parameter of the distribution.
        max_duration : int
            The maximum duration (range) for which the PDF is computed.

        Returns
        -------
        np.ndarray
            A 1D array representing the probability mass function for the distribution
            with the given shape and scale parameters, normalized to sum to 1.
        """
        p = self.distribution.pdf(np.arange(max_duration), shape, scale=scale)
        p = p / np.sum(p)
        p[np.isnan(p)] = 0  # remove potential nans
        return p

    def group_constructor(  # noqa: PLR0912
        self,
        trial_data: TrialData,
        grouping_dict: dict,
        channel_map: np.ndarray = None,
        time_map: np.ndarray = None,
        verbose: bool = False
    ) -> tuple[int, np.ndarray, dict]:
        """
        Adapt the model to groups by constructing group mappings and validating provided maps.

        Parameters
        ----------
        trial_data : TrialData
            The trial data containing trial-group information.
        grouping_dict : dict
            A dictionary defining groups for grouping modeling. Keys are group names,
            and values are lists of groups.
        channel_map : np.ndarray, optional
            A 2D array mapping channel parameters to groups. Default is None.
        time_map : np.ndarray, optional
            A 2D array mapping time parameters to groups. Default is None.
        verbose : bool, optional
            If True, prints detailed information about the group construction process.
            Default is False.

        Returns
        -------
        n_groups : int
            The number of unique groups.
        groups : np.ndarray
            An array indicating the group assignment for each trial.
        glabels : dict
            A dictionary containing group names and their corresponding modalities.
        """
        ## groups
        assert isinstance(grouping_dict, dict), "groups have to be specified as a dictionary"
        if len(grouping_dict.keys()) == 0:
            verbose = False
        # collect group names, groups, and trial coding
        group_names = []
        group_mods = []
        group_trials = []
        for group, mod in grouping_dict.items():
            group_names.append(group)
            group_mods.append(mod)
            group_trials.append(trial_data.xrdurations.coords[group])
            if verbose:
                print('group "' + group_names[-1] + '" analyzed, with groups:', group_mods[-1])

        group_mods = list(product(*group_mods))
        group_mods = np.array(group_mods, dtype=object)
        n_groups = len(group_mods)

        # build group array with digit indicating the combined groups
        if n_groups > 1:
            group_trials = np.vstack(group_trials).T
            groups = np.zeros((group_trials.shape[0])) * np.nan
            if verbose:
                print("\nCoded as follows: ")
            for i, mod in enumerate(group_mods):
                # assert len(np.where((group_trials == mod).all(axis=1))[0]) > 0, (
                #     f"Modality {mod} of group does not occur in the data"
                # )
                groups[np.where((group_trials == mod).all(axis=1))] = i
                if verbose:
                    print(str(i) + ": " + str(mod))
        else:
            groups = np.zeros(trial_data.n_trials)
        groups = np.int8(groups)
        glabels = {"group " + str(group_names): group_mods}

        # check maps if provided
        if channel_map is not None and time_map is not None:
            n_groups_mags = 0 if channel_map is None else channel_map.shape[0]
            n_groups_pars = 0 if time_map is None else time_map.shape[0]
            if (
                n_groups_mags > 0 and n_groups_pars > 0
            ):  # either both maps should have the same number of groups, or 0
                assert n_groups_mags == n_groups_pars, (
                    "Channel and time parameter maps have to indicate the same number of groups"
                )
                # make sure nr of events correspond per row
                for cur_group in range(n_groups):
                    assert (sum(channel_map[cur_group, :] >= 0) + 1
                            == sum(time_map[cur_group, :] >= 0)), (
                        "nr of events in channel map and time map do not correspond on row "
                        + str(cur_group)
                    )
            elif n_groups_mags == 0:
                assert not (time_map < 0).any(), (
                    "If negative time parameter are provided, channel map is required."
                )
                channel_map = np.zeros((n_groups, time_map.shape[1] - 1), dtype=int)
            else:
                time_map = np.zeros((n_groups, channel_map.shape[1] + 1), dtype=int)
                if (channel_map < 0).any():
                    for cur_group in range(n_groups):
                        time_map[cur_group, np.where(channel_map[cur_group, :] < 0)[0]] = -1
                        time_map[cur_group, np.where(channel_map[cur_group, :] < 0)[0] + 1] = 1

            # at this point, all should indicate the same number of groups
            assert n_groups == channel_map.shape[0] == time_map.shape[0], (
                "number of unique groups should correspond to number of rows in map(s)"
            )

            if verbose:
                print("\nChannel map:")
                for cnt in range(n_groups):
                    print(str(cnt) + ": ", channel_map[cnt, :])

                print("\nTime map:")
                for cnt in range(n_groups):
                    print(str(cnt) + ": ", time_map[cnt, :])

            # at this point, all should indicate the same number of groups
            assert n_groups == channel_map.shape[0] == time_map.shape[0], (
                "number of unique groups should correspond to number of rows in map(s)"
            )

        return n_groups, groups, glabels
