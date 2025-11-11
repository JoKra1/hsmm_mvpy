"""Transforms epoched data using a the ProjIdentity matrix for HMP analysis.

Returns electrode values after performing transformation steps without the projection. For consistency the channel dimension is renamed 'component'
"""

import numpy as np
import xarray as xr
from warnings import warn
from typing import Optional
from hmp.transformers.base import BaseTransformer

class ProjIdentity(BaseTransformer):
    """Transforms epoched data using a the ProjIdentity matrix for HMP analysis.

    Returns electrode values after performing transformation steps without the projection. For consistency the channel dimension is renamed 'component'
    
    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    interval_id: str
        Name of the variable that contains the per-trial intervals in the epoch_data used for cropping.
    offset_after : float
        Time offset after interval start for cropping.
    too_short : float, optional
        Minimum duration threshold for keeping epochs.
    too_long : float, optional
        Maximum duration threshold for keeping epochs.
    reject_threshold : float, optional
        Threshold for rejecting noisy epochs.
    common_variance : bool
        Whether to standardize variance across participants.
    whiten :bool
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
        interval_id: str = 'rt',
        offset_after: float = 0,
        too_short: Optional[float] = None,
        too_long: Optional[float] = None,
        reject_threshold: Optional[float] = None,
        common_variance: bool = False,
        whiten: bool = True,
        center: bool = False,
        copy: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            interval_id=interval_id,
            offset_after=offset_after,
            too_short=too_short,
            too_long=too_long,
            reject_threshold=reject_threshold,
            verbose=verbose,
            common_variance=common_variance,
            whiten=whiten,
            center=center,
            copy=copy,
        )
        warn('Identity projection might pose problems of dimensionality'
             'and collinearity of channels. Thus rendering HMP estimation'
             'difficult, use with care!')

        # Preprocessing
        data = self.common_preprocess(epoch_data)

        # Projection
        data = data.rename({"channel": "component"})
        data["component"] = np.arange(len(data.component))
        weights = np.identity(len(data.component))

        # Final formatting
        self.data = self.data_format(
            data, weights
        )
        self.weights = weights
        