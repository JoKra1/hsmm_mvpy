"""Transforms epoched data using a custom linear combination for HMP analysis.

Projects channels into a custom space given by a n_chan x n_component matrix. See example matrix when applying `ProjPCA` class.
"""
import xarray as xr
from warnings import warn
from typing import Optional
from hmp.transformers.base import BaseTransformer


class ProjCustom(BaseTransformer):
    """Transforms epoched data using a a custom linear combination for HMP analysis.

    Projects channels into a custom space given by a n_chan x n_component matrix. See example matrix when applying `ProjPCA` class.
    
    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    weights: xr.DataArray
        Custom linear combination of channels as an xarray.DataArray with 'channel' and 'component' dimensions and the weights
    interval_id: str
        Name of the variable that contains the per-trial intervals in the epoch_data used for cropping.
    offset_after_end : float
        Time offset after interval start for cropping.
    min_duration : float, optional
        Minimum duration threshold for keeping epochs.
    max_duration : float, optional
        Maximum duration threshold for keeping epochs.
    reject_threshold : float, optional
        Threshold for rejecting noisy epochs.
    common_variance : bool
        Whether to standardize variance across participants.
    whiten : bool
        Z-scoring the components from the projection to represent them all as de-meaned and at unit-variance
    center : bool
        Whether to center the data across the last dimension before projection
    copy : bool
        Whether to copy the data before transforming.
    verbose : bool
        Whether to print rejection/cropping details.
    """
    
    def __init__(
        self,
        epoch_data: xr.Dataset,
        weights: xr.DataArray,
        interval_id: str = 'rt',
        offset_after_end: float = 0,
        min_duration: float = 0,
        max_duration: float = float('Inf'),
        reject_threshold: Optional[float] = None,
        common_variance: bool = False,
        whiten: bool = True,
        center: bool = True,
        copy: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            interval_id=interval_id,
            offset_after_end=offset_after_end,
            min_duration=min_duration,
            max_duration=max_duration,
            reject_threshold=reject_threshold,
            verbose=verbose,
            common_variance=common_variance,
            whiten=whiten,
            center=center,
            copy=copy,
        )
        # Preprocessing
        data = self.common_preprocess(epoch_data)

        # Projection
        data = data @ weights

        # Final formatting
        self.data = self.data_format(data, weights)
        self.weights = weights
