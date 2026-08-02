"""Sampling-based estimation of HMP parameters with PyMC.

The likelihood is the same one EM uses, reimplemented in JAX so it can be
differentiated, and wrapped as a PyTensor Op. See
:mod:`hmp.estimators.jax_likelihood` and :mod:`hmp.estimators.pytensor_op`.

The scales are sampled on a log scale, which keeps them positive without a hard
boundary for the sampler to run into. The shape of the duration distribution is
held at the model's value rather than sampled, following the parametrisation in
the paper.

The priors are placeholders. They are weakly informative and centred on the
data, which is enough to sample, but they have not been agreed as the priors for
the method.
"""

import numpy as np

from hmp.estimators.base import BaseEstimator, EstimationResult
from hmp.estimators.pytensor_op import build_op


class MCMCEstimator(BaseEstimator):
    """Estimate HMP parameters by sampling the posterior.

    Parameters
    ----------
    draws : int, optional
        Posterior draws per chain. Default is 1000.
    tune : int, optional
        Tuning steps per chain, discarded. Default is 1000.
    chains : int, optional
        Number of chains. Default is 4.
    target_accept : float, optional
        NUTS target acceptance rate. Default is 0.9.
    channel_prior_sd : float, optional
        Standard deviation of the normal prior on channel contributions. If
        None, taken from the scale of the cross-correlated data.
    nuts_sampler : str, optional
        Passed to ``pm.sample``. "numpyro" compiles the whole graph to JAX and
        is much faster; "pymc" uses the default backend and the gradient Op.
    random_seed : int, optional
        Seed for sampling.
    progressbar : bool, optional
        Show the sampling progress bar. Default is False.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        channel_prior_sd: float = None,
        nuts_sampler: str = "numpyro",
        random_seed: int = None,
        progressbar: bool = False,
    ):
        super().__init__()
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.channel_prior_sd = channel_prior_sd
        self.nuts_sampler = nuts_sampler
        self.random_seed = random_seed
        self.progressbar = progressbar
        self.idata = None

    def supports_uncertainty(self) -> bool:
        """Report that this estimator provides uncertainty."""
        return True

    def build_model(self, model, pattern_data, groups=None):
        """Build the PyMC model for a set of HMP parameters.

        Parameters sharing an entry in the model's channel or time map are one
        random variable indexed into several positions, rather than several
        variables reconciled afterwards. The sampler therefore moves in the free
        parameter space and the sharing holds by construction, which projecting
        a proposal onto the constraint would not achieve.

        Returns
        -------
        pymc.Model
        """
        import pymc as pm  # noqa: PLC0415
        import pytensor.tensor as pt  # noqa: PLC0415

        n_dims = np.asarray(pattern_data.cross_corr).shape[1]
        shape = float(model.distribution.shape)

        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        n_groups = channel_map.shape[0]

        if n_groups > 1 and (channel_map.min() < 0 or time_map.min() < 0):
            raise NotImplementedError(
                "Groups that omit events are not supported yet; every entry of "
                "channel_map and time_map must be non-negative."
            )

        if groups is None:
            groups = np.zeros(len(pattern_data.durations), dtype=int)
        groups = np.asarray(groups)

        subsets = [groups == group for group in range(n_groups)]
        ops = [build_op(pattern_data, model, subset=subset) for subset in subsets]

        channel_sd = self.channel_prior_sd
        if channel_sd is None:
            channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) or 1.0

        # centre the scale prior where an even split of a trial would put it
        mean_duration = float(np.mean(np.concatenate([op.durations for op in ops])))
        even_scale = float(
            model.distribution.mean_to_scale(mean_duration / time_map.shape[1])
        )

        channel_index, n_channel_shared, _ = self._tie_index(channel_map)
        time_index, n_time_shared, _ = self._tie_index(time_map)

        with pm.Model() as pymc_model:
            # one variable per distinct parameter, indexed into position
            shared_channel = pm.Normal(
                "channel_pars", mu=0.0, sigma=channel_sd,
                shape=(n_channel_shared, n_dims),
            )
            shared_log_scale = pm.Normal(
                "log_scale", mu=np.log(even_scale), sigma=1.0, shape=n_time_shared
            )
            shared_scale = pm.Deterministic("scale", pt.exp(shared_log_scale))

            total = 0
            for group in range(n_groups):
                channel_group = shared_channel[channel_index[group]]
                scale_group = shared_scale[time_index[group]]
                time_group = pt.stack(
                    [pt.fill(scale_group, shape), scale_group], axis=1
                )  # shape of the duration distribution is held, not sampled
                total = total + ops[group](channel_group, time_group)

            pm.Potential("hmp_likelihood", total)

        self._even_scale = even_scale
        return pymc_model

    def fit(  # noqa: PLR0913, PLR0917
        self,
        model,
        pattern_data,
        initial_channel_pars: np.ndarray,
        initial_time_pars: np.ndarray,
        groups: np.ndarray = None,
        cpus: int = 1,  # noqa: ARG002
    ) -> EstimationResult:
        """Sample the posterior and summarise it as an EstimationResult.

        The starting points supplied by the model become the initial values of
        the chains, one per chain, recycled if there are more chains than
        starting points.

        Returns
        -------
        EstimationResult
            ``channel_pars`` and ``time_pars`` hold the posterior mean, shaped
            as the model expects. ``uncertainty`` holds posterior standard
            deviations, and ``diagnostics["idata"]`` the full InferenceData.
        """
        import pymc as pm  # noqa: PLC0415

        pymc_model = self.build_model(model, pattern_data, groups)

        initvals = self._initial_values(model, initial_channel_pars, initial_time_pars)

        with pymc_model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                nuts_sampler=self.nuts_sampler,
                initvals=initvals,
                random_seed=self.random_seed,
                progressbar=self.progressbar,
            )

        self.idata = idata
        self.fitted = True
        return self._summarise(idata, model)

    def posterior_event_probabilities(  # noqa: PLR0913, PLR0917
        self,
        model,
        pattern_data,
        n_draws: int = 100,
        groups: np.ndarray = None,
        random_seed: int = None,
    ):
        """Event probabilities implied by each of a sample of posterior draws.

        A fit gives one set of by-trial event probabilities, at one set of
        parameters. Sampling gives a distribution over them, which carries the
        parameter uncertainty into the quantity the method exists to estimate.

        Computed on request rather than during sampling, because the result is
        ``(draws, trials, samples, events)`` and only some of it is usually
        wanted.

        Parameters
        ----------
        model : EventModel
        pattern_data : PatternData
        n_draws : int, optional
            How many posterior draws to evaluate, sampled without replacement
            from all chains. Default is 100.
        groups : np.ndarray, optional
            Passed through to the model.
        random_seed : int, optional
            Seed for choosing which draws to use.

        Returns
        -------
        (n_draws, n_trials, n_samples, n_events) ndarray
        """
        if self.idata is None:
            raise ValueError("Nothing sampled yet, call fit first.")

        posterior = self.idata.posterior
        channel_draws = posterior["channel_pars"].stack(pooled=("chain", "draw"))
        scale_draws = posterior["scale"].stack(pooled=("chain", "draw"))
        available = channel_draws.sizes["pooled"]

        rng = np.random.default_rng(random_seed)
        chosen = rng.choice(available, size=min(n_draws, available), replace=False)

        shape = float(model.distribution.shape)
        channel_index, _, _ = self._tie_index(
            np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        )
        time_index, _, _ = self._tie_index(
            np.atleast_2d(np.asarray(model.time_map)).astype(int)
        )

        probabilities = []
        for index in chosen:
            shared_channel = np.asarray(
                channel_draws.isel(pooled=index).values, dtype=np.float64
            )
            shared_scale = np.asarray(
                scale_draws.isel(pooled=index).values, dtype=np.float64
            )
            # expand the shared values back out to one entry per group
            channel_pars = shared_channel[channel_index]
            scale = shared_scale[time_index]
            time_pars = np.stack([np.full(scale.shape, shape), scale], axis=-1)
            probabilities.append(
                model.event_probabilities(
                    pattern_data, channel_pars, time_pars, groups
                ).values
            )
        return np.stack(probabilities)

    @staticmethod
    def _tie_index(codes):
        """Give every distinct parameter in a map its own index.

        Codes are compared down each column, not across the whole map: groups
        carrying the same code at a given event share that event's parameter,
        while the same code at a different event is a different parameter. This
        matches how the maps are read when parameters are tied during EM.

        Returns
        -------
        index : (n_groups, n_positions) ndarray
            Where each group and position reads its value from.
        n_distinct : int
        first_seen : list of (group, position)
            Where each distinct parameter first appears.
        """
        codes = np.asarray(codes)
        index = np.zeros(codes.shape, dtype=int)
        first_seen = []
        for position in range(codes.shape[1]):
            column = codes[:, position]
            for code in np.unique(column):
                shared_by = column == code
                index[shared_by, position] = len(first_seen)
                first_seen.append((int(np.argmax(shared_by)), position))
        return index, len(first_seen), first_seen

    def _initial_values(self, model, initial_channel_pars, initial_time_pars):
        """One starting point per chain, taken from what the model generated.

        The starting points are laid out per group, while the variables are per
        distinct code, so each shared value is read from the first position
        carrying that code.
        """
        channel = np.asarray(initial_channel_pars, dtype=np.float64)
        time = np.asarray(initial_time_pars, dtype=np.float64)
        n_available = min(len(channel), len(time))

        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        _, _, channel_at = self._tie_index(channel_map)
        _, _, time_at = self._tie_index(time_map)

        initvals = []
        for chain in range(self.chains):
            source = chain % n_available
            channel_start = np.stack(
                [channel[source][group][event] for group, event in channel_at]
            )
            scale_start = np.array(
                [time[source][group][stage][1] for group, stage in time_at]
            )
            initvals.append(
                {
                    "channel_pars": channel_start,
                    "log_scale": np.log(np.clip(scale_start, 1e-6, None)),
                }
            )
        return initvals

    def _summarise(self, idata, model):
        """Posterior mean as the point estimate, with diagnostics alongside."""
        import arviz as az  # noqa: PLC0415

        posterior = idata.posterior
        shared_channel = posterior["channel_pars"].mean(dim=("chain", "draw")).values
        shared_scale = posterior["scale"].mean(dim=("chain", "draw")).values
        shape = float(model.distribution.shape)

        # expand the shared values back out to one entry per group
        channel_map = np.atleast_2d(np.asarray(model.channel_map)).astype(int)
        time_map = np.atleast_2d(np.asarray(model.time_map)).astype(int)
        channel_index, _, _ = self._tie_index(channel_map)
        time_index, _, _ = self._tie_index(time_map)
        channel_mean = shared_channel[channel_index]
        scale_mean = shared_scale[time_index]
        time_mean = np.stack(
            [np.full(scale_mean.shape, shape), scale_mean], axis=-1
        )

        variables = ["channel_pars", "scale"]
        summary = az.summary(idata, var_names=variables)
        # from the diagnostics themselves rather than from the summary table,
        # which rounds to two decimals and so cannot be compared against 1.01
        rhat = az.rhat(idata, var_names=variables)
        ess = az.ess(idata, var_names=variables)
        max_rhat = float(max(float(rhat[name].max()) for name in variables))
        min_ess = float(min(float(ess[name].min()) for name in variables))
        divergences = int(idata.sample_stats["diverging"].sum())

        shared_channel_sd = posterior["channel_pars"].std(dim=("chain", "draw")).values
        shared_scale_sd = posterior["scale"].std(dim=("chain", "draw")).values

        return EstimationResult(
            channel_pars=channel_mean,
            time_pars=time_mean,
            likelihood=float(np.max(idata.sample_stats["lp"].values)),
            converged=bool(max_rhat < 1.01 and divergences == 0),
            n_iterations=int(self.draws * self.chains),
            diagnostics={
                "idata": idata,
                "max_rhat": max_rhat,
                "min_ess": min_ess,
                "divergences": divergences,
                "summary": summary,
            },
            uncertainty={
                "channel_pars_sd": shared_channel_sd[channel_index],
                "scale_sd": shared_scale_sd[time_index],
            },
        )
