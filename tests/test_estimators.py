"""Tests for the parameter estimation architecture."""

import numpy as np
import pytest

from hmp.estimators import EMEstimator, EstimationResult


class TestEstimationResult:
    """Test EstimationResult class."""

    def test_creation(self):
        """Test basic EstimationResult creation."""
        result = EstimationResult(
            channel_pars=np.random.randn(2, 5),
            time_pars=np.random.rand(3, 2) + 0.1,
            likelihood=-100.5,
            converged=True,
            n_iterations=25,
            diagnostics={"method": "test"}
        )

        assert result.likelihood == -100.5
        assert result.converged is True
        assert result.n_iterations == 25
        assert result.channel_pars.shape == (2, 5)
        assert result.time_pars.shape == (3, 2)
        assert result.diagnostics["method"] == "test"


class TestEMEstimator:
    """Test EMEstimator class."""

    def test_creation(self):
        """Test EMEstimator creation with parameters."""
        em_est = EMEstimator(max_iteration=500, tolerance=1e-5, min_iteration=5)

        assert em_est.max_iteration == 500
        assert em_est.tolerance == 1e-5
        assert em_est.min_iteration == 5
        assert em_est.get_method_name() == "EMEstimator"
        assert em_est.supports_uncertainty() is False

    def test_default_parameters(self):
        """Test EMEstimator with default parameters."""
        em_est = EMEstimator()

        assert em_est.max_iteration == 1000
        assert em_est.tolerance == 1e-4
        assert em_est.min_iteration == 1


if __name__ == "__main__":
    pytest.main([__file__])
