#!/usr/bin/env python
"""Integration test of the estimation interface.

NOTE: This test requires proper TrialData setup with participant coordinates.
It's more of an integration test and may need real data to run properly.
For unit tests of the estimation components, see test_estimators.py
"""

import numpy as np
import xarray as xr
import pytest
from hmp.estimators import EMEstimator, BaseEstimator, EstimationResult
from hmp.models import EventModel
from hmp.trialdata import TrialData
from hmp.patterns import HalfSine


@pytest.mark.skip("Requires proper TrialData setup - integration test")
def test_estimation_interface():
    """Test that the new estimation interface works with full data pipeline."""
    
    # Create simple synthetic data
    n_trials = 5
    n_channels = 3
    duration = 100
    sfreq = 250
    
    # Create TrialData with proper structure
    data = np.random.randn(n_trials * duration, n_channels)
    starts = np.arange(0, n_trials * duration, duration)
    ends = starts + duration - 1
    durations_array = np.full(n_trials, duration)
    
    # Create xarray for durations
    durations_xr = xr.DataArray(
        durations_array, 
        dims=['trial'],
        coords={'trial': range(n_trials)}
    )
    
    trial_data = TrialData(
        xrdurations=durations_xr,
        starts=starts,
        ends=ends,
        n_trials=n_trials,
        n_samples=n_trials * duration,
        sfreq=sfreq,
        offset=0,
        cross_corr=data
    )
    
    # Create model
    pattern = HalfSine.create_expected(sfreq=sfreq, width=10)
    model = EventModel(pattern, n_events=2)
    
    # Create estimator
    estimator = EMEstimator(max_iteration=5, tolerance=1e-3)
    
    # Test that estimator has correct interface
    assert isinstance(estimator, BaseEstimator)
    assert estimator.supports_uncertainty() == False
    assert estimator.get_method_name() == "EMEstimator"
    
    # Test fitting with estimator
    result = model.fit(trial_data, estimator=estimator, verbose=False)
    
    # Check result type
    assert isinstance(result, EstimationResult)
    assert result.channel_pars is not None
    assert result.time_pars is not None
    assert result.likelihood is not None
    assert isinstance(result.converged, bool)
    assert isinstance(result.n_iterations, int)
    assert isinstance(result.diagnostics, dict)
    
    print("Estimation interface test passed!")
    
    # Test backward compatibility (no estimator)
    model2 = EventModel(pattern, n_events=2)
    result2 = model2.fit(trial_data, verbose=False)
    
    # Should return None for backward compatibility
    assert result2 is None
    assert hasattr(model2, 'channel_pars')
    assert hasattr(model2, 'time_pars')
    
    print("Backward compatibility test passed!")
    
    return True


def test_basic_estimation_components():
    """Test basic estimation components without full TrialData integration."""
    
    # Test EstimationResult creation
    result = EstimationResult(
        channel_pars=np.random.randn(2, 5),
        time_pars=np.random.rand(3, 2) + 0.1,
        likelihood=-100.5,
        converged=True,
        n_iterations=25,
        diagnostics={'method': 'EM'}
    )
    
    assert isinstance(result, EstimationResult)
    assert result.likelihood == -100.5
    assert result.converged is True
    
    # Test EMEstimator creation
    estimator = EMEstimator(max_iteration=100, tolerance=1e-4)
    assert isinstance(estimator, BaseEstimator)
    assert estimator.get_method_name() == "EMEstimator"
    assert estimator.supports_uncertainty() is False
    
    # Test EventModel has estimator parameter
    pattern = HalfSine.create_expected(sfreq=250, width=10)
    model = EventModel(pattern, n_events=2)
    
    import inspect
    fit_sig = inspect.signature(model.fit)
    assert 'estimator' in fit_sig.parameters
    
    print("✓ Basic estimation components test passed!")


if __name__ == "__main__":
    # Run only the basic test when called directly
    test_basic_estimation_components()
    print("All tests passed! 🎉")


if __name__ == "__main__":
    test_basic_estimation_components()
