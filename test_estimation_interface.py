#!/usr/bin/env python
"""Quick test of the estimation interface integration."""

import numpy as np
from hmp.estimators import EMEstimator, BaseEstimator, EstimationResult
from hmp.models import EventModel
from hmp.trialdata import TrialData
from hmp.patterns import HalfSine


def test_estimation_interface():
    """Test that the new estimation interface works."""
    
    # Create simple synthetic data
    n_trials = 5
    n_channels = 3
    duration = 100
    sfreq = 250
    
    # Create TrialData
    data = np.random.randn(n_trials * duration, n_channels)
    starts = np.arange(0, n_trials * duration, duration)
    ends = starts + duration - 1
    durations = np.full(n_trials, duration)
    
    trial_data = TrialData(
        cross_corr=data,
        starts=starts,
        ends=ends,
        durations=durations
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
    
    print("✓ Estimation interface test passed!")
    
    # Test backward compatibility (no estimator)
    model2 = EventModel(pattern, n_events=2)
    result2 = model2.fit(trial_data, verbose=False)
    
    # Should return None for backward compatibility
    assert result2 is None
    assert hasattr(model2, 'channel_pars')
    assert hasattr(model2, 'time_pars')
    
    print("✓ Backward compatibility test passed!")
    
    return True


if __name__ == "__main__":
    test_estimation_interface()
    print("All tests passed! 🎉")
