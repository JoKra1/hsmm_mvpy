"""Classes for transforming data to HMP format.

These provide methods to:

    1. Trim the trials data from epoch time 0 (e.g. stimulus) up to a specified interval
       (e.g. response), optionally including a fixed offset after the interval.
    2. Epochs whose interval exceeds lower and upper interval limits
       (`min_duration` and `max_duration`) are rejected.
       Additional rejection can be applied based on amplitude thresholds with `reject_threshold`
    3. Optionally standardize variance across subjects and center the data
    4. Project channels to new virtual channel, using the different classes,
        either based on a PCA (`ProjPCA`),
        arbitrary linear combination of channels (`ProjArbitrary`)
        or the identity of the channels (`ProjIdentity`)
    5. Whiten the components


Classes
-------
ProjPCA
    Project channels into principal component space based on the covariance matrix among electrodes
ProjArbitrary
    Apply a user-defined linear combination of original channels to a new set of virtual channels
ProjIdentity
    Returns the channels in the same space
"""

from abc import ABC
from typing import Optional
from warnings import warn

import numpy as np
import xarray as xr


class BaseTransformer(ABC):
    """Base class for HMP transformer pipelines.

    This class provides common transformer steps.

    Methods
    -------
    common_preprocess(data)
        Apply core transformer steps including rejection, variance standardization, and centering.

    reject_crop_epochs(data)
        Crop each epoch from time 0 of the epoch to its interval with optional rejection criteria.
        For epoch in the `epoch_data` xr.Dataset, this function trims the epoch data from epoch
        time 0 (e.g. stimulus onset) up to the specified interval (e.g. response), optionally
         including a fixed offset after the interval.
        Epochs whose interval exceeds specified lower and upper limits are rejected.
        Additional rejection can be applied based on signal amplitude thresholds in the interval.

    data_format(data, weights, transformer_model, ori_coords, sfreq, offset)
        Finalize the transformation by formatting the data.

    """

    def __init__(
            self,
            interval_id: str,
            offset_after_end: float,
            min_duration: Optional[float],
            max_duration: Optional[float],
            reject_threshold: Optional[float],
            verbose: bool,
            common_variance: bool,
            whiten: bool,
            center: bool,
            copy: bool,

    ):
        self.interval_id = interval_id
        self.offset_after_end = offset_after_end
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.reject_threshold = reject_threshold
        self.verbose = verbose
        self.common_variance = common_variance
        self.whiten = whiten
        self.center = center
        self.copy = copy

        if self.max_duration < 0 or self.min_duration < 0 or self.max_duration < self.min_duration:
            raise ValueError("Limit to intervals cannot be negative")

    def common_preprocess(self, epoch_data) -> xr.DataArray:
        self.sfreq = epoch_data.sfreq
        data = epoch_data.data.copy(deep=self.copy) if self.copy else epoch_data.data

        if self.common_variance:
            pstds = data.std(['epoch','sample','channel'], skipna=True)
        
        data = data.stack(trial=["participant", "epoch"]).dropna("trial", how="all")
        data = data.transpose('trial','channel','sample')

        if self.interval_id is not None:
            if self.max_duration is float('Inf'):
                self.max_duration = (int(epoch_data.sample.max()) - self.offset_after_end\
                                     * self.sfreq)/self.sfreq
            if self.min_duration == 0:
                self.min_duration = 1 / self.sfreq
            data = self.reject_crop_epochs(data)
        else:
            # removes baseline
            data = data.sel(sample = slice(0, int(data.sample.max())+1), drop=True).\
                    stack(trial=["participant", "epoch"])
            data = data.transpose("trial", "channel", "sample")
            warn('No intervals provided, fitting HMP on the whole epoch duration from center event')
            if self.reject_threshold is not np.inf or self.reject_threshold is not None:
                warn("No rejection threshold can be applied when no intervals are provided")
        
        if self.center:
            data -= data.mean(['sample'], skipna=True)

        if self.common_variance:
            data = data.unstack()
            data /= pstds
            data = data.stack(trial=["participant", "epoch"])

        return data.dropna("trial", how="all")

    def reject_crop_epochs(self, epoch_data:xr.Dataset):
        """
        Crop each epoch from time 0 of the epoch to its interval with optional rejection criteria.

        For epoch in the `epoch_data` xr.Dataset, this function trims the epoch data from epoch
         time 0 (e.g. stimulus onset) up to the specified interval (e.g. response), optionally
         including a fixed offset after the interval.
        Epochs whose interval exceeds specified lower and upper limits are rejected, and
         additional rejection can be applied based on signal amplitude thresholds in the interval.

        Returns
        -------
        epoch_data : np.ndarray
            Array of cropped epoch data that passed all criteria.
        """
        rts_arr = epoch_data.coords[self.interval_id].values.copy()
        max_rt = np.nanmax(rts_arr)
        if max_rt > 500:
            warn(f"Found intervals with a max value value of {np.round(max_rt,2)}"
                ", assuming intervals are in milliseconds and converting to seconds")
            rts_arr /= 1000
        offset_after_end_samples = int(np.rint(self.offset_after_end * self.sfreq))

        rts_arr[rts_arr > self.max_duration] = 0
        rts_arr[rts_arr < self.min_duration] = 0
        rt_criteria_rej = len(rts_arr[rts_arr == 0])
        inexistant_rej = len(rts_arr[np.isnan(rts_arr)])
        rts_arr[np.isnan(rts_arr)] = 0  # rejected during epoching or inexistant
        # Converting to samples
        rts_arr = np.rint(rts_arr * self.sfreq).astype(int)

        epoch_data = epoch_data.sel(
                    sample=slice(0, 
                        min(
                            int(rts_arr.max() + offset_after_end_samples + 1),
                            epoch_data.sample.max().values
                        )
                ), drop=True
        )
        
        if len(rts_arr[rts_arr > 0]) == 0:
            raise ValueError("No intervals are between the requested limits of "\
                f"minimum {self.min_duration} and maximum {self.max_duration} seconds")

        min_rt = min(rts_arr[rts_arr > 0])
        if min_rt < 10:
            warn("The shortest interval is less than 10 samples. "
                 "Consider specifying too short trials using the `min_duration` parameter "
                 "or increasing sampling frequency of the signal.")

        if self.verbose:
            print(f"{len(rts_arr[rts_arr > 0])} intervals between {self.min_duration} and "\
                f"{self.max_duration} seconds.")
        rej = 0
        reject_threshold = self.reject_threshold if self.reject_threshold is not None else np.inf
        for i in range(len(epoch_data.data)):
            if rts_arr[i] > 0:
                # Crops the epochs up to duration
                if (
                    np.abs(epoch_data.values[i, :, :rts_arr[i] +\
                        offset_after_end_samples])
                    < (reject_threshold)
                ).all():
                    epoch_data.values[i, :, rts_arr[i] + offset_after_end_samples:] = np.nan
                elif ~np.isnan(epoch_data.values[i, :, 0]).any():
                    epoch_data.values[i, :, :] = np.nan
                    rej += 1
                else: # assumes rejected before
                    epoch_data.values[i, :, :] = np.nan
                    inexistant_rej += 1
            else:
                epoch_data.values[i, :, :] = np.nan

        if rej == len(epoch_data):
            raise ValueError('All trials rejected, inspect intervals and rejection criterion')
        if self.verbose:
            print(f"Rejection summary: \n {rej} trials rejected based on threshold of "
             f"{self.reject_threshold} \n {rt_criteria_rej} trials rejected based on interval "
             f"limit of {self.min_duration, self.max_duration} \n {inexistant_rej} trials "
             "detected with no interval (e.g. preprocessing or interval exceeding epoch)) ")
        return epoch_data

    @staticmethod
    def compute_covariance(data, center):
        vcov_mat = np.zeros((data.sizes["channel"], data.sizes["channel"]), dtype=np.float64)
        count = 0
        for i in data.trial:
            x_i = np.squeeze(data.sel(trial=i).values)
            x_i = x_i[:, ~np.isnan(x_i[0, :])]
            if x_i.shape[1] > x_i.shape[0]:
                if center:
                    cov_i = (x_i @ x_i.T)/(x_i.shape[1]-1)
                else:
                    cov_i = np.cov(x_i, bias=1)
                vcov_mat += cov_i
                count += 1
        vcov_mat /= count
        return vcov_mat

    def data_format(
        self,
        data: xr.DataArray,
        weights: xr.DataArray
        ) -> xr.DataArray:
        """Finalize the transformation: transpose, reassign coords, stack, and store attributes."""
        data.attrs["sfreq"] = self.sfreq
        data.attrs["offset"] = self.offset_after_end
        comp_stdev = data.std(['trial','sample'], skipna=True)
        if self.whiten:
            data /= comp_stdev
        self.data = data
        self.weights = weights
        return self.data, comp_stdev
