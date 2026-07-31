"""Tests for the estimation interface."""

import numpy as np
import pytest
from test_io import init_data

import hmp
from hmp.estimators import BaseEstimator, EMEstimator, EstimationResult
from hmp.models import EventModel
from hmp.patterndata import PatternData


def data():
    """Noiseless single-participant data, as used by the other model tests."""
    event_b, _, epoch_data, _, _, n_events = init_data()
    hmp_data = hmp.basedata.default(
        epoch_data, n_comp=3, center=True, duration_id="response_time"
    )
    pdata_b = PatternData.from_basedata(hmp_data.select_coord("b", "subject"))
    return event_b, epoch_data, pdata_b, n_events


class MockEstimator(BaseEstimator):
    """Minimal estimator used to check that injection is honored."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_called = False

    def fit(self, model, pattern_data, initial_channel_pars, initial_time_pars,  # noqa: ARG002
            groups=None, cpus=1):  # noqa: ARG002
        """Return the starting point untouched, without running any estimation."""
        self.fit_called = True
        self.fitted = True
        return EstimationResult(
            channel_pars=initial_channel_pars[0],
            time_pars=initial_time_pars[0],
            likelihood=100.0,
            converged=True,
            n_iterations=0,
            diagnostics={
                "traces": np.array([100.0]),
                "traces_group": np.array([[100.0]]),
                "time_pars_dev": initial_time_pars[:1],
            },
        )


class TestEstimationResult:
    """The container passed back from every estimator."""

    def test_creation(self):
        result = EstimationResult(
            channel_pars=np.array([[1, 2], [3, 4]]),
            time_pars=np.array([[0.5, 1.0]]),
            likelihood=100.0,
            converged=True,
            n_iterations=10,
            diagnostics={"traces": np.array([1.0])},
        )
        assert result.likelihood == 100.0
        assert result.converged is True
        assert result.n_iterations == 10
        assert result.uncertainty is None

    def test_uncertainty_is_optional(self):
        """Bayesian estimators may attach uncertainty; EM leaves it unset."""
        result = EstimationResult(
            channel_pars=np.array([[1, 2]]),
            time_pars=np.array([[0.5, 1.0]]),
            likelihood=50.0,
            converged=False,
            n_iterations=5,
            uncertainty={"channel_std": np.array([[0.1, 0.2]])},
        )
        assert "channel_std" in result.uncertainty
        assert result.diagnostics == {}


class TestBaseEstimator:
    """The abstract interface all estimation methods implement."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseEstimator()

    def test_subclass_interface(self):
        estimator = MockEstimator()
        assert not estimator.is_fitted
        assert estimator.get_method_name() == "MockEstimator"
        assert not estimator.supports_uncertainty()


class TestEMEstimator:
    """EMEstimator construction."""

    def test_defaults(self):
        em = EMEstimator()
        assert em.tolerance == 1e-4
        assert em.max_iteration == 1e3
        assert em.min_iteration == 1
        assert em.n_cor == 30
        assert not em.is_fitted
        assert em.get_method_name() == "EMEstimator"
        assert not em.supports_uncertainty()

    def test_custom(self):
        em = EMEstimator(tolerance=1e-3, max_iteration=500, min_iteration=5, n_cor=10)
        assert em.tolerance == 1e-3
        assert em.max_iteration == 500
        assert em.min_iteration == 5
        assert em.n_cor == 10


class TestEstimatorInjection:
    """EventModel.fit delegates the estimation to a swappable estimator."""

    def test_injected_em_matches_default(self):
        """Passing an equivalent EMEstimator must not change the fit at all.

        This is the guard on the refactor: routing the estimation through
        EMEstimator has to reproduce the in-model implementation exactly.
        """
        _, _, pdata_b, n_events = data()

        default_model = EventModel(n_events=n_events)
        default_model.fit(pdata_b, verbose=False)

        injected_model = EventModel(n_events=n_events)
        injected_model.fit(pdata_b, verbose=False, estimator=EMEstimator())

        assert injected_model.lkhs == default_model.lkhs
        assert np.array_equal(injected_model.time_pars, default_model.time_pars)
        assert np.array_equal(injected_model.channel_pars, default_model.channel_pars)

    def test_estimator_settings_change_the_fit(self):
        """A coarser estimator must actually stop earlier.

        Without this, `test_injected_em_matches_default` would also pass if the
        estimator argument were silently ignored.
        """
        _, _, pdata_b, n_events = data()

        fine = EventModel(n_events=n_events)
        fine.fit(pdata_b, verbose=False)

        coarse = EventModel(n_events=n_events)
        coarse.fit(pdata_b, verbose=False,
                   estimator=EMEstimator(tolerance=1e-1, max_iteration=2))

        assert len(coarse.traces) < len(fine.traces)

    def test_custom_estimator_is_used(self):
        """A non-EM estimator drives the fit and populates the model."""
        _, _, pdata_b, n_events = data()

        model = EventModel(n_events=n_events)
        mock = MockEstimator()
        result = model.fit(pdata_b, verbose=False, estimator=mock)

        assert mock.fit_called
        assert mock.is_fitted
        assert model._fitted
        assert result.likelihood == 100.0
        assert model.lkhs == result.likelihood
        assert np.array_equal(model.channel_pars, result.channel_pars)
        assert np.array_equal(model.time_pars, result.time_pars)

    def test_result_is_returned_and_stored(self):
        _, _, pdata_b, n_events = data()

        model = EventModel(n_events=n_events)
        result = model.fit(pdata_b, verbose=False)

        assert isinstance(result, EstimationResult)
        assert model.estimation_result is result
        assert result.converged
        assert result.n_iterations >= 1

    def test_diagnostics_structure(self):
        _, _, pdata_b, n_events = data()

        model = EventModel(n_events=n_events)
        result = model.fit(pdata_b, verbose=False)

        for key in ("traces", "traces_group", "time_pars_dev", "lkhs"):
            assert key in result.diagnostics

        traces = result.diagnostics["traces"]
        assert isinstance(traces, np.ndarray)
        assert len(traces) == result.n_iterations + 1
        assert result.diagnostics["time_pars_dev"].shape[1:] == result.time_pars.shape


class TestBackwardCompatibility:
    """Existing entry points keep working through the estimator."""

    def test_fit_transform(self):
        _, _, pdata_b, n_events = data()

        model = EventModel(n_events=n_events)
        lkh, estimates = model.fit_transform(pdata_b, verbose=False)

        assert isinstance(lkh, (float, np.floating, np.ndarray))
        assert hasattr(estimates, "dims")

    def test_multiple_starting_points(self):
        """Selection across starting points is the estimator's responsibility.

        Note `_format_parameters` builds ``starting_points + 1`` time parameter
        sets but only ``starting_points`` channel parameter sets, so pairing
        them drops the last proposal. That is pre-existing model behaviour,
        kept as-is here; the estimator simply evaluates every pair it is given.
        """
        _, _, pdata_b, n_events = data()

        model = EventModel(n_events=n_events, starting_points=3)
        result = model.fit(pdata_b, verbose=False)

        assert len(result.diagnostics["lkhs"]) == model.starting_points
        assert result.likelihood == np.max(result.diagnostics["lkhs"])


if __name__ == "__main__":
    pytest.main([__file__])
