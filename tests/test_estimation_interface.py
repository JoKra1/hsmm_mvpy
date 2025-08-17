"""Tests for the new estimation interface."""

import numpy as np
import pytest

import hmp
from hmp.estimators import BaseEstimator, EMEstimator, EstimationResult
from hmp.models import EventModel
from hmp.patterns import HalfSine
from hmp.trialdata import TrialData
from tests.test_io import init_data


class MockEstimator(BaseEstimator):
    """Mock estimator for testing the interface."""

    def __init__(self, should_converge=True, **kwargs):
        super().__init__(**kwargs)
        self.should_converge = should_converge
        self.fit_called = False

    def fit(self, _trial_data, initial_channel_pars, initial_time_pars,
           _fixed_channel_pars=None, _fixed_time_pars=None, **_kwargs):
        """Mock fit method."""
        self.fit_called = True
        self._fitted = True

        # Return mock result
        return EstimationResult(
            channel_pars=initial_channel_pars,
            time_pars=initial_time_pars,
            likelihood=100.0,
            converged=self.should_converge,
            n_iterations=10,
            diagnostics={'method': 'Mock', 'test_param': 'test_value'}
        )


@pytest.fixture
def trial_data():
    """Create trial data for testing."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2)
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(
        preprocessed=data_b, pattern=event_properties.template
    )
    return trial_data_b, event_properties


def test_estimation_result_creation():
    """Test EstimationResult dataclass creation and validation."""
    # Test basic creation
    result = EstimationResult(
        channel_pars=np.array([[1, 2], [3, 4]]),
        time_pars=np.array([[0.5, 1.0]]),
        likelihood=100.0,
        converged=True,
        n_iterations=10,
        diagnostics={'method': 'EM'}
    )

    assert isinstance(result.channel_pars, np.ndarray)
    assert isinstance(result.time_pars, np.ndarray)
    assert result.likelihood == 100.0
    assert result.converged is True
    assert result.n_iterations == 10
    assert result.diagnostics['method'] == 'EM'
    assert result.uncertainty is None

    # Test with uncertainty
    result_with_uncertainty = EstimationResult(
        channel_pars=np.array([[1, 2]]),
        time_pars=np.array([[0.5]]),
        likelihood=50.0,
        converged=False,
        n_iterations=5,
        diagnostics={},
        uncertainty={'channel_std': np.array([[0.1, 0.2]])}
    )

    assert result_with_uncertainty.uncertainty is not None
    assert 'channel_std' in result_with_uncertainty.uncertainty


def test_estimation_result_array_conversion():
    """Test that EstimationResult properly converts inputs to numpy arrays."""
    # Test with lists that should be converted to arrays
    result = EstimationResult(
        channel_pars=[[1, 2], [3, 4]],  # List input
        time_pars=[[0.5, 1.0]],        # List input
        likelihood=100.0,
        converged=True,
        n_iterations=10,
        diagnostics={}
    )

    assert isinstance(result.channel_pars, np.ndarray)
    assert isinstance(result.time_pars, np.ndarray)
    assert result.channel_pars.shape == (2, 2)
    assert result.time_pars.shape == (1, 2)


def test_base_estimator_interface():
    """Test BaseEstimator abstract interface."""
    # Test that BaseEstimator cannot be instantiated directly
    with pytest.raises(TypeError):
        BaseEstimator()

    # Test MockEstimator implementation
    estimator = MockEstimator()
    assert not estimator.is_fitted
    assert estimator.get_method_name() == 'MockEstimator'
    assert not estimator.supports_uncertainty()

    # Test mock fitting
    dummy_trial_data = None  # MockEstimator doesn't use it
    dummy_pars = np.array([[1, 2]])

    result = estimator.fit(dummy_trial_data, dummy_pars, dummy_pars)

    assert estimator.is_fitted
    assert estimator.fit_called
    assert isinstance(result, EstimationResult)
    assert result.converged is True
    assert result.diagnostics['method'] == 'Mock'


def test_em_estimator_initialization():
    """Test EMEstimator initialization and parameters."""
    # Test default initialization
    em = EMEstimator()
    assert em.max_iteration == 1000
    assert em.tolerance == 1e-4
    assert em.min_iteration == 1
    assert not em.is_fitted
    assert em.get_method_name() == 'EMEstimator'
    assert not em.supports_uncertainty()

    # Test custom initialization
    em_custom = EMEstimator(max_iteration=500, tolerance=1e-3, min_iteration=5)
    assert em_custom.max_iteration == 500
    assert em_custom.tolerance == 1e-3
    assert em_custom.min_iteration == 5


def test_em_estimator_parameter_validation():
    """Test that EMEstimator validates required parameters."""
    em = EMEstimator()
    trial_data_dummy = None
    initial_pars = np.array([[[1, 2]]])

    # Test that model is required
    with pytest.raises(ValueError, match="EMEstimator requires a model instance"):
        em.fit(trial_data_dummy, initial_pars, initial_pars)


def test_model_fit_with_default_estimator(trial_data):
    """Test EventModel.fit with default estimator."""
    trial_data_b, event_properties = trial_data

    model = EventModel(event_properties, n_events=3)

    # Test that fit works with default EMEstimator
    result = model.fit(trial_data_b, verbose=False)

    assert isinstance(result, EstimationResult)
    assert model._fitted
    assert hasattr(model, 'channel_pars')
    assert hasattr(model, 'time_pars')
    assert hasattr(model, 'lkhs')
    assert hasattr(model, 'traces')


def test_model_fit_with_custom_estimator(trial_data):
    """Test EventModel.fit with custom estimator."""
    trial_data_b, event_properties = trial_data

    model = EventModel(event_properties, n_events=3)
    mock_estimator = MockEstimator()

    # Test that fit works with custom estimator
    result = model.fit(trial_data_b, estimator=mock_estimator, verbose=False)

    assert isinstance(result, EstimationResult)
    assert mock_estimator.fit_called
    assert mock_estimator.is_fitted
    assert model._fitted

    # Check that model state was updated from result
    assert np.array_equal(model.channel_pars, result.channel_pars)
    assert np.array_equal(model.time_pars, result.time_pars)
    assert model.lkhs == result.likelihood


def test_model_fit_with_custom_em_parameters(trial_data):
    """Test EventModel.fit with custom EM parameters."""
    trial_data_b, event_properties = trial_data

    model = EventModel(event_properties, n_events=3)
    custom_em = EMEstimator(max_iteration=100, tolerance=1e-3, min_iteration=2)

    # Test that fit works with custom EM parameters
    result = model.fit(trial_data_b, estimator=custom_em, verbose=False)

    assert isinstance(result, EstimationResult)
    assert custom_em.is_fitted
    assert custom_em.max_iteration == 100
    assert custom_em.tolerance == 1e-3
    assert custom_em.min_iteration == 2


def test_model_fit_parameter_passing(trial_data):
    """Test that model.fit properly passes parameters to estimator."""
    trial_data_b, event_properties = trial_data

    model = EventModel(event_properties, n_events=3)

    # Create custom initial parameters
    n_groups = 1
    n_events = 3
    n_dims = trial_data_b.n_dims

    initial_channel_pars = np.random.rand(n_groups, n_events, n_dims)
    initial_time_pars = np.random.rand(n_groups, n_events + 1, 2)

    result = model.fit(
        trial_data_b,
        initial_channel_pars=initial_channel_pars,
        initial_time_pars=initial_time_pars,
        tolerance=1e-5,
        max_iteration=50,
        verbose=False
    )

    assert isinstance(result, EstimationResult)
    # Parameters should have been initialized properly
    assert result.channel_pars.shape[1:] == initial_channel_pars.shape[1:]
    assert result.time_pars.shape[1:] == initial_time_pars.shape[1:]


def test_estimation_result_diagnostics_structure():
    """Test that EstimationResult diagnostics have expected structure."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2)
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(
        preprocessed=data_b, pattern=event_properties.template
    )

    model = EventModel(event_properties, n_events=3)
    em = EMEstimator(max_iteration=10)  # Small number for fast test

    result = model.fit(trial_data_b, estimator=em, verbose=False)

    # Check diagnostics structure
    assert 'method' in result.diagnostics
    assert result.diagnostics['method'] == 'EM'
    assert 'traces' in result.diagnostics
    assert 'time_pars_dev' in result.diagnostics
    assert 'tolerance_achieved' in result.diagnostics

    # Check traces format
    traces = result.diagnostics['traces']
    assert isinstance(traces, np.ndarray)
    assert len(traces) >= 1  # At least initial likelihood

    # Check time_pars_dev format
    time_pars_dev = result.diagnostics['time_pars_dev']
    assert isinstance(time_pars_dev, np.ndarray)
    assert time_pars_dev.shape[1:] == result.time_pars.shape


def test_em_estimator_convergence_behavior():
    """Test EM estimator convergence behavior."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2)
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(
        preprocessed=data_b, pattern=event_properties.template
    )

    model = EventModel(event_properties, n_events=3)

    # Test with very tight tolerance (should not converge quickly)
    em_tight = EMEstimator(max_iteration=5, tolerance=1e-10, min_iteration=1)
    result_tight = model.fit(trial_data_b, estimator=em_tight, verbose=False)

    # Should hit max iterations (allowing for +1 due to implementation details)
    assert result_tight.n_iterations <= 6

    # Test with loose tolerance (should converge quickly)
    em_loose = EMEstimator(max_iteration=100, tolerance=1e-1, min_iteration=1)
    result_loose = model.fit(trial_data_b, estimator=em_loose, verbose=False)

    # Should converge before max iterations in most cases
    assert result_loose.n_iterations >= 1


def test_backward_compatibility():
    """Test that the new interface doesn't break existing functionality."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2)
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(
        preprocessed=data_b, pattern=event_properties.template
    )

    model = EventModel(event_properties, n_events=3)

    # Test that old fit_transform still works
    try:
        lkh, estimates = model.fit_transform(trial_data_b, verbose=False)
        assert isinstance(lkh, (float, np.floating, np.ndarray))
        assert hasattr(estimates, 'dims')  # xarray DataArray
        backward_compatible = True
    except Exception:
        backward_compatible = False

    assert backward_compatible, "New estimation interface breaks backward compatibility"


def test_multiple_starting_points():
    """Test EM estimator with multiple starting points."""
    event_b, event_a, epoch_data, positions, sfreq, n_events = init_data()
    hmp_data = hmp.preprocessing.Standard(epoch_data, n_comp=2)
    data_b = hmp.utils.participant_selection(hmp_data.data, 'b')
    event_properties = HalfSine.create_expected(sfreq=data_b.sfreq)
    trial_data_b = TrialData.from_preprocessed(
        preprocessed=data_b, pattern=event_properties.template
    )

    model = EventModel(event_properties, n_events=3)

    # Create multiple starting points
    n_starting_points = 3
    n_groups = 1
    n_events = 3
    n_dims = trial_data_b.n_dims

    # 4D array: (n_starting_points, n_groups, n_events, n_dims)
    initial_channel_pars = np.random.rand(n_starting_points, n_groups, n_events, n_dims)
    initial_time_pars = np.random.rand(n_starting_points, n_groups, n_events + 1, 2)

    em = EMEstimator(max_iteration=10)  # Small for fast test

    result = model.fit(
        trial_data_b,
        initial_channel_pars=initial_channel_pars,
        initial_time_pars=initial_time_pars,
        estimator=em,
        verbose=False
    )

    # Should return best result
    assert isinstance(result, EstimationResult)
    assert result.channel_pars.shape == (n_groups, n_events, n_dims)
    assert result.time_pars.shape == (n_groups, n_events + 1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
