"""Markov Chain Monte Carlo estimator for HMP models."""

from warnings import warn
import numpy as np
import warnings
from typing import Optional, Dict, Any

try:
    from .base import BaseEstimator, EstimationResult
except ImportError:
    # Fallback for direct imports
    from base import BaseEstimator, EstimationResult
from hmp.trialdata import TrialData

try:
    import pymc as pm
    import arviz as az
    import pytensor
    import pytensor.tensor as at
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None
    pytensor = None
    at = None

try:
    from .jax_ops_simple import JAXHMPLikelihoodOp, check_jax_available
    JAX_OP_AVAILABLE = check_jax_available() and PYMC_AVAILABLE and JAXHMPLikelihoodOp is not None
except ImportError:
    JAX_OP_AVAILABLE = False
    JAXHMPLikelihoodOp = None


class MCMCEstimator(BaseEstimator):
    """MCMC parameter estimator using PyMC.
    
    This estimator provides Bayesian parameter estimation for HMP models
    using Markov Chain Monte Carlo sampling. It provides uncertainty
    quantification through posterior distributions.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of MCMC samples to draw. Default is 2000.
    n_tune : int, optional
        Number of tuning samples. Default is 1000.
    n_chains : int, optional
        Number of parallel chains. Default is 4.
    target_accept : float, optional
        Target acceptance rate for NUTS. Default is 0.8.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    prior_config : dict, optional
        Prior configuration for parameters. Default uses weakly informative priors.
    step_method : str, optional
        Sampling method to use. Options: 'auto', 'nuts', 'metropolis', 'slice'.
        'auto' uses NUTS if gradients available, otherwise Metropolis. Default is 'auto'.
    use_gradients : bool, optional
        Whether to use gradient information for sampling. Default is True.
        When False, uses gradient-free samplers explicitly.
    gradient_eps : float, optional
        Step size for finite difference gradient computation. Default is 1e-6.
        Only used when gradients are computed numerically.
    """
    
    def __init__(self, n_samples: int = 2000, n_tune: int = 1000, n_chains: int = 4,
                 target_accept: float = 0.8, random_seed: Optional[int] = None,
                 prior_config: Optional[Dict[str, Any]] = None, 
                 step_method: str = 'auto', use_gradients: bool = True,
                 gradient_eps: float = 1e-6, **kwargs):
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required for MCMCEstimator. Install with: pip install pymc")
            
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.prior_config = prior_config or {}
        self.step_method = step_method
        self.use_gradients = use_gradients
        self.gradient_eps = gradient_eps
        self.trace = None
        self.model = None
        
        # Validate step_method parameter
        valid_methods = ['auto', 'nuts', 'metropolis', 'slice']
        if step_method not in valid_methods:
            raise ValueError(f"step_method must be one of {valid_methods}, got '{step_method}'")
        
    def fit(self, trial_data, initial_channel_pars: np.ndarray, 
           initial_time_pars: np.ndarray, model=None, verbose: bool = True, 
           fixed_channel_pars: Optional[list] = None,
           fixed_time_pars: Optional[list] = None, **kwargs) -> EstimationResult:
        """Fit model parameters using MCMC.
        
        Parameters
        ----------
        trial_data : TrialData
            The trial data to fit the model to
        initial_channel_pars : np.ndarray
            Initial channel parameter values (used for initialization)
        initial_time_pars : np.ndarray
            Initial time distribution parameter values (used for initialization)
        model : EventModel
            The model instance (needed for likelihood computation)
        verbose : bool, optional
            Whether to print verbose output. Default is True.
        fixed_channel_pars : list, optional
            List of channel parameter indices to fix during estimation. 
            Not currently implemented. Default is None.
        fixed_time_pars : list, optional
            List of time parameter indices to fix during estimation.
            Use [0] to fix shape parameters, [1] to fix scale parameters,
            or [0, 1] to fix both. Default is None (estimate all parameters).
        **kwargs
            Additional fitting options (currently unused)
            
        Returns
        -------
        EstimationResult
            Results of the MCMC estimation with uncertainty estimates
        """
        if model is None:
            raise ValueError("MCMCEstimator requires a model instance for likelihood computation")
        if not isinstance(trial_data, TrialData):
            trial_data = TrialData.from_preprocessed(trial_data, model.template)
            
        # Store verbose setting for use in step method selection
        self._verbose = verbose
            
        if initial_time_pars.ndim > 2:
            if initial_time_pars.shape[0] > 1:
                warn("MCMC estimator currently only supports single-group models. Using parameters from the first group.")
            initial_time_pars = initial_time_pars[0]

        # Replace NaNs in initial_time_pars with default values and ensure positivity
        for i in range(initial_time_pars.shape[0]):
            if np.isnan(initial_time_pars[i, 0]) or initial_time_pars[i, 0] <= 0:
                initial_time_pars[i, 0] = 2.0  # shape parameter
            if np.isnan(initial_time_pars[i, 1]) or initial_time_pars[i, 1] <= 0:
                initial_time_pars[i, 1] = 10.0  # scale parameter
        
        # Ensure time parameters are positive (required for gamma distribution)
        initial_time_pars = np.maximum(initial_time_pars, 1e-6)
        if initial_channel_pars.ndim > 2:
            if initial_channel_pars.shape[0] > 1:
                warn("MCMC estimator currently only supports single-group models. Using parameters from the first group.")
            initial_channel_pars = initial_channel_pars[0]
        
        initial_channel_pars = np.nan_to_num(initial_channel_pars)
        n_events = initial_channel_pars.shape[-2]
        n_dims = initial_channel_pars.shape[-1]
        n_stages = initial_time_pars.shape[-2]
        
        # Store for JAX implementation
        self._n_events = n_events
        
        # Build the PyMC model
        with pm.Model() as pymc_model:
            self.model = pymc_model
            
            # Channel parameter priors
            channel_pars = pm.Normal(
                "channel_pars", 
                mu=self.prior_config.get("channel_mu", 0.0),
                sigma=self.prior_config.get("channel_sigma", 1.0),
                shape=(n_events, n_dims),
                initval=initial_channel_pars
            )
            
            # Time parameter priors
            if fixed_time_pars is not None and 0 in fixed_time_pars:
                shape_pars = pm.Deterministic("shape_pars", at.as_tensor_variable(initial_time_pars[..., 0]))
            else:
                # Shape parameters (must be positive)
                shape_pars = pm.HalfNormal(
                    "shape_pars",
                    sigma=self.prior_config.get("shape_sigma", 2.0),
                    shape=n_stages,
                    initval=initial_time_pars[..., 0]
                )

            if fixed_time_pars is not None and 1 in fixed_time_pars:
                scale_pars = pm.Deterministic("scale_pars", at.as_tensor_variable(initial_time_pars[..., 1]))
            else:
                # Scale parameters (must be positive)
                scale_pars = pm.HalfNormal(
                    "scale_pars", 
                    sigma=self.prior_config.get("scale_sigma", 5.0),
                    shape=n_stages,
                    initval=initial_time_pars[..., 1]
                )
            
            # Combine time parameters
            time_pars = pm.Deterministic(
                "time_pars",
                pm.math.stack([shape_pars, scale_pars], axis=-1)
            )
            
            # Combine channel and time parameters into a single flattened variable
            flat_pars = pm.Deterministic(
                "flat_pars",
                pm.math.concatenate([channel_pars.flatten(), time_pars.flatten()])
            )

            # Custom likelihood based on HMP forward-backward algorithm
            likelihood = pm.Potential(
                "likelihood",
                self._log_likelihood_func(trial_data, model, channel_pars, time_pars)
            )
            
            # Determine step method and sample from posterior
            step = self._get_step_method(model)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                self.trace = pm.sample(
                    draws=self.n_samples,
                    tune=self.n_tune,
                    chains=self.n_chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=True,
                    step=step
                )
        
        # Extract results
        posterior = self.trace.posterior
        
        # Point estimates (posterior means)
        channel_pars_mean = posterior["channel_pars"].mean(dim=["chain", "draw"]).values
        time_pars_mean = posterior["time_pars"].mean(dim=["chain", "draw"]).values
        
        # Compute final likelihood using the posterior means
        final_likelihood, _ = model.estim_probs(trial_data, channel_pars_mean, time_pars_mean)
        
        # Convergence diagnostics
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        
        # Check convergence
        converged = (rhat["channel_pars"].max() < 1.1).values
        if "shape_pars" in rhat:
            converged = converged and (rhat["shape_pars"].max() < 1.1).values
        if "scale_pars" in rhat:
            converged = converged and (rhat["scale_pars"].max() < 1.1).values
        converged = bool(converged)
        
        # Uncertainty estimates
        uncertainty = {
            "channel_pars_std": posterior["channel_pars"].std(dim=["chain", "draw"]).values,
            "time_pars_std": posterior["time_pars"].std(dim=["chain", "draw"]).values,
            "channel_pars_quantiles": {
                "5%": posterior["channel_pars"].quantile(0.05, dim=["chain", "draw"]).values,
                "95%": posterior["channel_pars"].quantile(0.95, dim=["chain", "draw"]).values,
            },
            "time_pars_quantiles": {
                "5%": posterior["time_pars"].quantile(0.05, dim=["chain", "draw"]).values,
                "95%": posterior["time_pars"].quantile(0.95, dim=["chain", "draw"]).values,
            },
            "posterior_samples": {
                "channel_pars": posterior["channel_pars"].values,
                "time_pars": posterior["time_pars"].values,
            }
        }
        
        # Diagnostics
        rhat_dict = {"channel_pars": rhat["channel_pars"].values}
        if "shape_pars" in rhat:
            rhat_dict["shape_pars"] = rhat["shape_pars"].values
        if "scale_pars" in rhat:
            rhat_dict["scale_pars"] = rhat["scale_pars"].values
            
        ess_dict = {"channel_pars": ess["channel_pars"].values}
        if "shape_pars" in ess:
            ess_dict["shape_pars"] = ess["shape_pars"].values
        if "scale_pars" in ess:
            ess_dict["scale_pars"] = ess["scale_pars"].values
            
        diagnostics = {
            "method": "MCMC",
            "step_method": self.step_method,
            "gradients_used": getattr(self, '_gradients_available', False),
            "rhat": rhat_dict,
            "ess": ess_dict,
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "trace": self.trace
        }
        
        self._fitted = True
        
        return EstimationResult(
            channel_pars=channel_pars_mean,
            time_pars=time_pars_mean,
            likelihood=final_likelihood,
            converged=converged,
            n_iterations=self.n_samples,
            diagnostics=diagnostics,
            uncertainty=uncertainty
        )
    
    def _get_step_method(self, model):
        """Determine appropriate step method based on configuration."""
        import warnings
        
        # Check if gradients are available for the likelihood
        has_gradients = self.use_gradients and hasattr(self, '_gradients_available') and self._gradients_available
        
        if self.step_method == 'auto':
            if self.use_gradients and has_gradients:
                if hasattr(self, '_verbose') and self._verbose:
                    gradient_method = "JAX" if JAX_OP_AVAILABLE else "PyTensor"
                    print(f"Using NUTS sampler with {gradient_method} gradients")
                return pm.NUTS(target_accept=self.target_accept)
            else:
                if hasattr(self, '_verbose') and self._verbose:
                    if not self.use_gradients:
                        reason = "gradients disabled"
                    elif not JAX_OP_AVAILABLE:
                        reason = "JAX not available, PyTensor gradients failed"
                    else:
                        reason = "gradients not available"
                    print(f"Using Metropolis sampler ({reason})")
                return pm.Metropolis()
        elif self.step_method == 'nuts':
            if self.use_gradients and has_gradients:
                return pm.NUTS(target_accept=self.target_accept)
            else:
                warnings.warn(
                    "NUTS requested but gradients not available. Falling back to Metropolis.",
                    UserWarning
                )
                return pm.Metropolis()
        elif self.step_method == 'metropolis':
            return pm.Metropolis()
        elif self.step_method == 'slice':
            return pm.Slice()
        else:
            # Should not happen due to validation in __init__, but just in case
            return pm.Metropolis()
    
    def _log_likelihood_func(self, trial_data, model, channel_pars, time_pars):
        """
        Returns a log-likelihood function for PyMC model.
        This function computes the HMP log-likelihood using either JAX-based 
        forward-backward algorithm (if available) or numpy fallback.
        """
        # Check if we should use JAX implementation
        use_jax = (
            self.use_gradients and 
            JAX_OP_AVAILABLE and 
            self._should_use_jax(trial_data)
        )
        
        if use_jax:
            try:
                result = self._log_likelihood_func_jax(trial_data, model, channel_pars, time_pars)
                if self._verbose:
                    print("Using JAX implementation with automatic gradients")
                return result
            except Exception as e:
                if self._verbose:
                    print(f"JAX implementation failed ({e}), falling back to numpy implementation")
                # Mark gradients as unavailable for this session
                self._gradients_available = False
                return self._log_likelihood_func_numpy(trial_data, model, channel_pars, time_pars)
        else:
            # Use numpy implementation
            if hasattr(self, '_verbose') and self._verbose:
                if not JAX_OP_AVAILABLE:
                    reason = "JAX not available"
                else:
                    reason = "using fallback"
                print(f"Using numpy implementation ({reason})")
            self._gradients_available = False
            return self._log_likelihood_func_numpy(trial_data, model, channel_pars, time_pars)
            
    def _should_use_jax(self, trial_data):
        """Determine if JAX should be used based on data characteristics."""
        # Always try JAX first - let the exception handling deal with failures
        return True
    
    def _log_likelihood_func_jax(self, trial_data, model, channel_pars, time_pars):
        """JAX-based log-likelihood function with automatic differentiation."""
        import pytensor.tensor as at
        
        # Prepare static data for JAX
        cross_corr = trial_data.cross_corr
        durations = trial_data.durations
        starts = trial_data.starts  
        ends = trial_data.ends
        n_trials = trial_data.n_trials
        n_samples = trial_data.n_samples
        n_dims = trial_data.n_dims
        
        # Get number of events from initial parameters shape (before PyTensor tensors)
        # We need to access this during model building, not during tensor computation
        if hasattr(self, '_n_events'):
            n_events = self._n_events
        else:
            # This will be set during the fit method before this function is called
            raise ValueError("Number of events not set. This should be set in fit method.")
        
        n_stages = n_events + 1
        locations = np.zeros(n_stages, dtype=int)
        if hasattr(model, 'location') and model.location is not None:
            locations[1:-1] = model.location
        
        # Create JAX Op and apply it
        jax_op = JAXHMPLikelihoodOp()
        likelihood_result = jax_op.make_node(
            channel_pars, time_pars,
            at.as_tensor_variable(cross_corr),
            at.as_tensor_variable(durations),
            at.as_tensor_variable(starts),
            at.as_tensor_variable(ends),
            at.as_tensor_variable(locations)
        ).outputs[0]
        
        # Mark gradients as available
        self._gradients_available = True
        
        return likelihood_result
    
    def _log_likelihood_func_numpy(self, trial_data, model, channel_pars, time_pars):
        """NumPy-based log-likelihood function (fallback when JAX is not available)."""
        import pytensor.tensor as at
        
        # Create a PyTensor Op that uses the original numpy implementation
        class NumpyHMPLikelihoodOp(at.Op):
            __props__ = ()
            
            def make_node(self, channel_pars, time_pars, cross_corr, durations, starts, ends, locations):
                # Convert inputs to PyTensor tensors
                channel_pars = at.as_tensor_variable(channel_pars)
                time_pars = at.as_tensor_variable(time_pars) 
                cross_corr = at.as_tensor_variable(cross_corr)
                durations = at.as_tensor_variable(durations)
                starts = at.as_tensor_variable(starts)
                ends = at.as_tensor_variable(ends)
                locations = at.as_tensor_variable(locations)
                
                inputs = [channel_pars, time_pars, cross_corr, durations, starts, ends, locations]
                outputs = [at.dscalar()]
                return at.Apply(self, inputs, outputs)
            
            def perform(self, node, inputs, outputs):
                channel_pars, time_pars, cross_corr, durations, starts, ends, locations = inputs
                
                # Use the original numpy implementation from the model
                likelihood, _ = model.estim_probs(
                    trial_data, channel_pars, time_pars, location=True
                )
                
                outputs[0][0] = np.asarray(likelihood, dtype=node.outputs[0].dtype)
        
        numpy_op = NumpyHMPLikelihoodOp()
        
        # Prepare static data (all as numpy arrays)
        cross_corr = trial_data.cross_corr
        durations = trial_data.durations if hasattr(trial_data.durations, 'values') else trial_data.durations
        if hasattr(durations, 'values'):
            durations = durations.values
        starts = trial_data.starts  
        ends = trial_data.ends
        
        # Get the shape of time_pars to create locations
        # Since time_pars is a PyTensor variable, we need to get its shape another way
        n_stages = model.n_events + 1  # This should be the number of stages
        locations = np.zeros(n_stages, dtype=int)  # Default locations
        
        # Create the PyTensor computation graph
        likelihood_result = numpy_op.make_node(
            channel_pars, time_pars, cross_corr, durations, starts, ends, locations
        ).outputs[0]
        
        # Mark that gradients are not available with numpy fallback
        self._gradients_available = False
        
        return likelihood_result
    

    def supports_uncertainty(self) -> bool:
        """MCMC provides uncertainty estimates through posterior distributions."""
        return True
        
    def get_posterior_samples(self, parameter: str = None) -> np.ndarray:
        """Get posterior samples for parameters.
        
        Parameters
        ----------
        parameter : str, optional
            Parameter name ('channel_pars', 'time_pars', or None for all)
            
        Returns
        -------
        np.ndarray or dict
            Posterior samples
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting posterior samples")
            
        if parameter is None:
            return {
                "channel_pars": self.trace.posterior["channel_pars"].values,
                "time_pars": self.trace.posterior["time_pars"].values
            }
        elif parameter in self.trace.posterior:
            return self.trace.posterior[parameter].values
        else:
            raise ValueError(f"Parameter '{parameter}' not found in trace")
    
    def plot_trace(self, parameter: str = None):
        """Plot MCMC traces for diagnostics.
        
        Parameters
        ----------
        parameter : str, optional
            Parameter to plot. If None, plots all parameters.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before plotting traces")
            
        if parameter is None:
            az.plot_trace(self.trace)
        else:
            az.plot_trace(self.trace, var_names=[parameter])
            
    def summary(self) -> str:
        """Get summary statistics of posterior distributions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting summary")
            
        return str(az.summary(self.trace))