"""Tests for the differentiable likelihood and the sampling estimator.

The likelihood is checked against reference values captured from the numpy
implementation in float64, stage by stage rather than only on the final number,
so a disagreement says where it is. Regenerate the reference with
``python tests/gen_data/generate_reference.py``.
"""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import hmp
from hmp.models import EventModel
from hmp.patterndata import PatternData
from test_io import init_data

REFERENCE = Path(__file__).parent / "gen_data" / "likelihood_reference.npz"
SHIFT = 1  # hmp.distributions.Gamma.shift

# Reference values are float64 throughout, so agreement is demanded at a
# tolerance set by double precision rather than by whatever happens to pass.
EXACT = 1e-10


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE.name} missing; run tests/gen_data/generate_reference.py")
    return np.load(REFERENCE, allow_pickle=False)


@pytest.fixture(scope="module")
def layout(reference):
    """Return the data and trial layout shared by every reference case."""
    durations = reference["durations"]
    stacked = np.vstack(
        [reference["cross_corr"][s:e + 1]
         for s, e in zip(reference["starts"], reference["ends"])]
    )
    return {
        "cross_corr": stacked,
        "trial_starts": np.concatenate([[0], np.cumsum(durations)[:-1]]),
        "durations": durations,
        "max_duration": int(durations.max()),
    }


def case_names():
    """List the reference cases to parametrise over."""
    if not REFERENCE.exists():
        return []
    return list(np.load(REFERENCE, allow_pickle=False)["cases"])


def scaled_error(actual, expected):
    """Return the error relative to the scale of the array.

    These arrays span hundreds of orders of magnitude, so an elementwise
    relative error on entries that are legitimately near zero means nothing.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.max(np.abs(expected))), 1e-300)
    return float(np.max(np.abs(actual - expected)) / scale)


class TestLikelihoodMatchesReference:
    """The JAX likelihood must reproduce the numpy one it was ported from."""

    @pytest.mark.parametrize("case", case_names())
    def test_stages_match(self, case, reference, layout):
        from hmp.estimators import jax_likelihood as jl

        channel_pars = reference[f"{case}__channel_pars"]
        time_pars = reference[f"{case}__time_pars"]
        locations = reference[f"{case}__locations_samples"]

        computed_gains = jl.gains(layout["cross_corr"], channel_pars)
        assert scaled_error(computed_gains, reference[f"{case}__gains"]) < EXACT

        pmf = jl.stage_pmf(time_pars, locations, layout["max_duration"], SHIFT)
        assert scaled_error(pmf, reference[f"{case}__pmf"]) < EXACT

        per_trial = jl.trial_log_likelihood(
            layout["cross_corr"], channel_pars, time_pars, layout["trial_starts"],
            layout["durations"], locations, layout["max_duration"], SHIFT,
        )
        assert scaled_error(per_trial, reference[f"{case}__trial_likelihood"]) < EXACT

    @pytest.mark.parametrize("case", case_names())
    def test_total_matches(self, case, reference, layout):
        from hmp.estimators import jax_likelihood as jl

        total = jl.log_likelihood(
            layout["cross_corr"], reference[f"{case}__channel_pars"],
            reference[f"{case}__time_pars"], layout["trial_starts"],
            layout["durations"], reference[f"{case}__locations_samples"],
            layout["max_duration"], SHIFT,
        )
        assert scaled_error(total, reference[f"{case}__likelihood"]) < EXACT


class TestGradients:
    """A wrong gradient can hide behind correct values, so check it separately.

    The density goes through log(0) at the origin, where jnp.where masks the
    value but not the gradient, so this is a live failure mode rather than a
    formality.
    """

    @pytest.mark.parametrize("case", case_names())
    def test_matches_finite_differences(self, case, reference, layout):
        from jax.test_util import check_grads

        from hmp.estimators import jax_likelihood as jl

        channel_pars = jnp.asarray(reference[f"{case}__channel_pars"])
        time_pars = jnp.asarray(reference[f"{case}__time_pars"])
        locations = jnp.asarray(reference[f"{case}__locations_samples"])

        def total(channel, time):
            return jl.log_likelihood(
                layout["cross_corr"], channel, time, layout["trial_starts"],
                layout["durations"], locations, layout["max_duration"], SHIFT,
            )

        grad_channel, grad_time = jax.grad(total, argnums=(0, 1))(channel_pars, time_pars)
        assert np.isfinite(np.asarray(grad_channel)).all()
        assert np.isfinite(np.asarray(grad_time)).all()

        check_grads(total, (channel_pars, time_pars), order=1, modes=["rev"],
                    atol=2e-4, rtol=2e-4)


@pytest.fixture(scope="module")
def fitted_setup():
    """Return float64 data and an EM fit, for the bridge and sampler tests."""
    _, _, epoch_data, _, _, n_events = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    pattern_data = PatternData.from_basedata(hmp_data, dtype=np.float64)
    fitted = EventModel(n_events=n_events)
    fitted.fit(pattern_data, verbose=False)
    return pattern_data, fitted, n_events


class TestPyTensorBridge:
    """The Op has to agree with the function it wraps, on both backends."""

    def test_backends_agree_with_numpy(self, fitted_setup):
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.tensor as pt

        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]

        # float64 parameters: EM stores channel_pars as float32, which caps
        # agreement at about 1e-7 for reasons unrelated to this implementation
        channel_pars = np.asarray(fitted.channel_pars, dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars, dtype=np.float64)
        expected = model.log_likelihood(pattern_data, channel_pars, time_pars)

        op = build_op(pattern_data, model)
        channel_input, time_input = pt.dmatrix("channel"), pt.dmatrix("time")
        out = op(channel_input, time_input)

        default = float(pytensor.function([channel_input, time_input], out)(
            channel_pars[0], time_pars[0]))
        as_jax = float(pytensor.function([channel_input, time_input], out, mode="JAX")(
            channel_pars[0], time_pars[0]))

        assert abs(default - expected) / abs(expected) < EXACT
        assert default == as_jax

    def test_gradient_available_on_both_backends(self, fitted_setup):
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.tensor as pt

        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        channel_input, time_input = pt.dmatrix("channel"), pt.dmatrix("time")
        gradients = pt.grad(op(channel_input, time_input), [channel_input, time_input])
        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars[0], dtype=np.float64)

        results = []
        for mode in (None, "JAX"):
            grad_channel, grad_time = pytensor.function(
                [channel_input, time_input], gradients, mode=mode
            )(channel_pars, time_pars)
            assert np.isfinite(grad_channel).all()
            assert np.isfinite(grad_time).all()
            results.append((grad_channel, grad_time))

        assert np.allclose(results[0][0], results[1][0], atol=0, rtol=EXACT)
        assert np.allclose(results[0][1], results[1][1], atol=0, rtol=EXACT)


class TestMCMCEstimator:
    """The PyMC model has to be wired to the likelihood we verified."""

    def test_model_logp_is_likelihood_plus_priors(self, fitted_setup):
        pytest.importorskip("pymc")
        from scipy import stats

        from hmp.estimators.mcmc import MCMCEstimator
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]

        estimator = MCMCEstimator()
        pymc_model = estimator.build_model(model, pattern_data)

        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        scale = np.asarray(fitted.time_pars[0], dtype=np.float64)[:, 1]
        total = float(pymc_model.compile_logp()(
            {"channel_pars": channel_pars, "log_scale": np.log(scale)}
        ))

        op = build_op(pattern_data, model)
        shape = float(model.distribution.shape)
        time_pars = np.column_stack([np.full(scale.shape, shape), scale])
        likelihood = float(op._call_jax(channel_pars, time_pars))

        channel_sd = float(np.std(np.asarray(pattern_data.cross_corr)))
        priors = (
            stats.norm.logpdf(channel_pars, 0.0, channel_sd).sum()
            + stats.norm.logpdf(np.log(scale), np.log(estimator._even_scale), 1.0).sum()
        )
        assert abs(total - (likelihood + priors)) / abs(likelihood + priors) < EXACT

    def test_samples_and_reports_diagnostics(self, fitted_setup):
        pytest.importorskip("pymc")
        pytest.importorskip("numpyro")

        from hmp.estimators.base import EstimationResult
        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        _, groups, _ = model.group_constructor(pattern_data.durations, verbose=False)
        channel_pars, time_pars = model._format_parameters(
            None, None, groups, 1, pattern_data.durations, pattern_data.sfreq
        )

        estimator = MCMCEstimator(draws=100, tune=100, chains=2, random_seed=0)
        result = estimator.fit(model, pattern_data, channel_pars, time_pars, groups=groups)

        assert isinstance(result, EstimationResult)
        assert estimator.supports_uncertainty()
        assert result.channel_pars.shape == (1, n_events, model.n_dims)
        assert result.time_pars.shape == (1, n_events + 1, 2)
        # the shape of the duration distribution is held, not sampled
        assert np.allclose(result.time_pars[0][:, 0], model.distribution.shape)
        for key in ("idata", "max_rhat", "min_ess", "divergences"):
            assert key in result.diagnostics
        assert set(result.uncertainty) == {"channel_pars_sd", "scale_sd"}

        # the reported r_hat has to be the diagnostic itself, not the value from
        # az.summary, which rounds to two decimals and so cannot be compared
        # against a 1.01 threshold
        import arviz as az

        variables = ["channel_pars", "scale"]
        rhat = az.rhat(result.diagnostics["idata"], var_names=variables)
        recomputed = max(float(rhat[name].max()) for name in variables)
        assert result.diagnostics["max_rhat"] == pytest.approx(recomputed, abs=1e-12)

    def test_grouped_models_are_refused(self, fitted_setup):
        pytest.importorskip("pymc")
        from hmp.estimators.mcmc import MCMCEstimator

        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        groups = np.arange(len(pattern_data.durations)) % 2

        with pytest.raises(NotImplementedError, match="single-group"):
            MCMCEstimator().fit(model, pattern_data, np.zeros((1, 1, n_events, 3)),
                                np.ones((1, 1, n_events + 1, 2)), groups=groups)


class TestParameterPrecision:
    """Why comparisons against the numpy likelihood have to use float64 parameters."""

    def test_float32_parameters_cap_agreement(self, fitted_setup):
        """EM stores channel_pars as float32, which limits agreement to about 1e-7.

        estim_probs evaluates ``channel_pars ** 2 / 2`` in the dtype of the
        array it is given, so float32 parameters make the whole likelihood
        float32-accurate even when the data is float64. This is a property of
        how the parameters are stored, not of either implementation.
        """
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        assert fitted.channel_pars.dtype == np.float32

        as_float64 = np.asarray(fitted.channel_pars, dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars, dtype=np.float64)
        reference = float(op._call_jax(as_float64[0], time_pars[0]))

        from_stored = model.log_likelihood(
            pattern_data, fitted.channel_pars, fitted.time_pars
        )
        from_float64 = model.log_likelihood(pattern_data, as_float64, time_pars)

        assert abs(reference - from_float64) / abs(from_float64) < EXACT
        assert abs(reference - from_stored) / abs(from_stored) > EXACT


class TestEMFixedPoint:
    """EM does not stop at a stationary point of the likelihood.

    Eq 10 recovers the scale from the probability-weighted mean interval using
    ``mean_to_scale``, which inverts the mean of the untruncated continuous
    gamma. Eq 3 uses a gamma discretised over ``0..max_duration`` and
    renormalised, whose mean is different once appreciable mass falls past the
    support. So the scale update is inconsistent with the distribution the
    likelihood actually uses.

    These tests record that, because it means the posterior mode and the EM
    solution should not be expected to coincide. If the update is ever made
    consistent, they should start failing.
    """

    def test_scale_update_assumes_a_mean_the_pmf_does_not_have(self, fitted_setup):
        pattern_data, _, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        durations = np.asarray(pattern_data.ends) - np.asarray(pattern_data.starts) + 1
        max_duration = int(durations.max())
        support = np.arange(max_duration)

        # a scale putting appreciable mass beyond the support
        scale = max_duration / 4
        pmf = model.distribution_pdf(model.distribution.shape, scale, max_duration)

        assumed = float(model.distribution.scale_to_mean(scale))
        actual = float(np.sum(support * pmf))
        assert abs(actual - assumed) / assumed > 0.05

    def test_likelihood_still_improves_at_the_em_solution(self, fitted_setup):
        from hmp.estimators.pytensor_op import build_op

        pattern_data, fitted, n_events = fitted_setup
        model = EventModel(n_events=n_events)
        model.n_dims = pattern_data.cross_corr.shape[1]
        op = build_op(pattern_data, model)

        channel_pars = np.asarray(fitted.channel_pars[0], dtype=np.float64)
        time_pars = np.asarray(fitted.time_pars[0], dtype=np.float64)
        grad_channel, grad_time = jax.grad(op._call_jax, argnums=(0, 1))(
            channel_pars, time_pars
        )
        grad_channel = np.asarray(grad_channel)
        grad_time = np.asarray(grad_time)

        # the channel contributions do reach a stationary point
        assert np.linalg.norm(grad_channel) < 1e-2
        # the scales do not, by orders of magnitude
        assert np.linalg.norm(grad_time[:, 1]) > 10 * np.linalg.norm(grad_channel)

        before = float(op._call_jax(channel_pars, time_pars))
        stepped = time_pars.copy()
        stepped[:, 1] = time_pars[:, 1] + 1e-2 * grad_time[:, 1]
        after = float(op._call_jax(channel_pars + 1e-2 * grad_channel, stepped))
        assert after > before


if __name__ == "__main__":
    pytest.main([__file__])
