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

    def build_model(self, model, pattern_data):
        """Build the PyMC model for a set of HMP parameters.

        Returns
        -------
        pymc.Model
        """
        import pymc as pm  # noqa: PLC0415
        import pytensor.tensor as pt  # noqa: PLC0415

        n_events = model.n_events
        n_stages = n_events + 1
        n_dims = np.asarray(pattern_data.cross_corr).shape[1]
        shape = float(model.distribution.shape)

        op = build_op(pattern_data, model)

        channel_sd = self.channel_prior_sd
        if channel_sd is None:
            channel_sd = float(np.std(np.asarray(pattern_data.cross_corr))) or 1.0

        # centre the scale prior where an even split of the trial would put it
        mean_duration = float(np.mean(op.durations))
        even_scale = float(model.distribution.mean_to_scale(mean_duration / n_stages))

        with pm.Model() as pymc_model:
            channel_pars = pm.Normal(
                "channel_pars", mu=0.0, sigma=channel_sd, shape=(n_events, n_dims)
            )
            log_scale = pm.Normal(
                "log_scale", mu=np.log(even_scale), sigma=1.0, shape=n_stages
            )
            scale = pm.Deterministic("scale", pt.exp(log_scale))

            time_pars = pt.stack(
                [pt.fill(scale, shape), scale], axis=1
            )  # (n_stages, 2), shape held fixed

            pm.Potential("hmp_likelihood", op(channel_pars, time_pars))

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

        if groups is not None and len(np.unique(groups)) > 1:
            raise NotImplementedError(
                "MCMCEstimator currently supports single-group models only."
            )

        pymc_model = self.build_model(model, pattern_data)

        initvals = self._initial_values(initial_channel_pars, initial_time_pars)

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

    def _initial_values(self, initial_channel_pars, initial_time_pars):
        """One starting point per chain, taken from what the model generated."""
        channel = np.asarray(initial_channel_pars, dtype=np.float64)
        time = np.asarray(initial_time_pars, dtype=np.float64)
        n_available = min(len(channel), len(time))

        initvals = []
        for chain in range(self.chains):
            source = chain % n_available
            scale = time[source][0][:, 1]
            initvals.append(
                {
                    "channel_pars": channel[source][0],
                    "log_scale": np.log(np.clip(scale, 1e-6, None)),
                }
            )
        return initvals

    def _summarise(self, idata, model):
        """Posterior mean as the point estimate, with diagnostics alongside."""
        import arviz as az  # noqa: PLC0415

        posterior = idata.posterior
        channel_mean = posterior["channel_pars"].mean(dim=("chain", "draw")).values
        scale_mean = posterior["scale"].mean(dim=("chain", "draw")).values
        shape = float(model.distribution.shape)

        time_mean = np.column_stack([np.full(scale_mean.shape, shape), scale_mean])

        variables = ["channel_pars", "scale"]
        summary = az.summary(idata, var_names=variables)
        # from the diagnostics themselves rather than from the summary table,
        # which rounds to two decimals and so cannot be compared against 1.01
        rhat = az.rhat(idata, var_names=variables)
        ess = az.ess(idata, var_names=variables)
        max_rhat = float(max(float(rhat[name].max()) for name in variables))
        min_ess = float(min(float(ess[name].min()) for name in variables))
        divergences = int(idata.sample_stats["diverging"].sum())

        return EstimationResult(
            # leading axis of 1 is the group dimension the model expects
            channel_pars=channel_mean[np.newaxis, ...],
            time_pars=time_mean[np.newaxis, ...],
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
                "channel_pars_sd": posterior["channel_pars"]
                .std(dim=("chain", "draw"))
                .values[np.newaxis, ...],
                "scale_sd": posterior["scale"].std(dim=("chain", "draw")).values,
            },
        )
