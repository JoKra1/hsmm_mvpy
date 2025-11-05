"""Classes for transforming data to HMP format.

These provide methods to:

    1 Trim the trials data from epoch time 0 (e.g. stimulus onset) up to the specified interval (e.g. response), optionally including a fixed offset after the interval.
    2. Optionally standardize variance across subjects and center the data
    3. Epochs whose interval exceeds specified lower and upper interval limits (`too_short` and `too_long` are rejected, additional rejection can be applied based on signal amplitude thresholds in the interval with `reject_threshold`
    4. Project channels to new virtual channel, using the different classes, either based on a PCA (`Standard`), MCCA (`MCCA_aligned`), arbitrary linear combination of channels (`Arbitrary`) or the identity of the channels (`Identity`)
    5. zscore the data for different levels depending on the dataset


Classes
-------
Standard
    Project channels into principal component space based on the covariance matrix among electrodes
Arbitrary
    Apply a user-defined linear combination of original channels to a new set of virtual channels
Identity
    Returns the channels in the same space
MCCA_aligned
    Per-subject PCA re-aligned into a new common space
"""

from enum import Enum
from typing import Optional, Union, Any
from warnings import warn
from abc import ABC, abstractmethod
from pandas import MultiIndex
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from hmp import mcca
from hmp.utils import reject_crop_epochs, zscore_xarray

class ApplyZScore(Enum):
    """Enum for different settings on applying the z-score.

    Possible values: [ALL, PARTICIPANT, TRIAL, DONT_APPLY]
    """

    ALL = 'all'
    PARTICIPANT = 'participant'
    TRIAL = 'trial'
    DONT_APPLY = 'dont_apply'

    def __str__(self) -> str:
        return self.value

    def __bool__(self) -> bool:
        return self != self.DONT_APPLY

    @classmethod
    def parse(cls, label):
        if isinstance(label, str):
            label = label.lower()

        if isinstance(label, cls):
            return label
        elif label in (False, None, 'dont_apply'):
            return cls.DONT_APPLY
        elif label in ('trial', True):
            return cls.TRIAL
        elif label == 'participant':
            return cls.PARTICIPANT
        elif label == 'all':
            return cls.ALL
        else:
            raise KeyError(f"Unknown value for apply_zscore: '{label}'; "
            f"valid options: [{', '.join([e.value for e in cls])}] or Bool (True defaults to {cls.TRIAL})")


class BasePreprocessing(ABC):
    """Base class for HMP preprocessing pipelines.

    This class provides common preprocessing steps.

    Methods
    -------

    common_preprocess()
        Apply core preprocessing steps including rejection, standardization, and centering.

    center_data(data)
        Center the data by subtracting the mean across the last dimension.

    stack_data(data)
        Stack data from [participant, epoch, sample, component] to [all_samples, component].

    standardize(x)
        Standardize participant data by scaling to group-level mean variance.

    zscoring(data)
        Apply z-scoring based on the configured mode (all, participant, trial).

    data_format(data, weights, preprocessing_model, ori_coords, sfreq, offset)
        Finalize the transformation by formatting and stacking the data.
    """

    def __init__(
            self,
            interval_id: str,
            offset_after: float,
            too_short: Optional[float],
            too_long: Optional[float],
            reject_threshold: Optional[float],
            verbose: bool,
            apply_standard: bool,
            apply_zscore: Union[bool, str, ApplyZScore],
            centering: bool,
            copy: bool,

    ):
        apply_zscore = ApplyZScore.parse(apply_zscore)        
        self.interval_id = interval_id
        self.offset_after = offset_after
        self.too_short = too_short
        self.too_long = too_long
        self.reject_threshold = reject_threshold
        self.verbose = verbose
        self.apply_standard = apply_standard
        self.apply_zscore = apply_zscore
        self.centering = centering
        self.copy = copy
        self.verbose = verbose
    
    def common_preprocess(self, epoch_data) -> xr.DataArray:
        self.sfreq = epoch_data.sfreq
        data = epoch_data.data.copy(deep=True) if self.copy else epoch_data.data

        if self.apply_standard:
            if "participant" not in data.dims or len(data.participant) == 1:
                warn("Standardization requested but participant dimension is missing or singular.")
            else:
                mean_std = data.groupby('participant').std(dim="channel").data.mean()
                data = data.assign(mean_std=mean_std.data)
                data = data.groupby('participant').map(self.standardize)

        if self.centering:
            data = self.center_data(data)
            
        data = reject_crop_epochs(data,
            self.sfreq,
            interval_id=self.interval_id,
            offset_after=self.offset_after,
            too_short=self.too_short,
            too_long=self.too_long,
            reject_threshold=self.reject_threshold,
            verbose=self.verbose
        )
        if isinstance(data, xr.Dataset):
            data = data.data
        if np.isnan(data.groupby("participant").mean(["epoch", "sample"]).values).any():
            raise ValueError("At least one participant has an empty channel")

        data = data.transpose("participant", "epoch", "channel", "sample")
        return data
    
    @staticmethod
    def center_data(data: xr.DataArray) -> xr.DataArray:
        """Center the data."""
        mean_last_dim = np.mean(data.values, axis=-1)
        mean_last_dim_expanded = np.expand_dims(mean_last_dim, axis=-1)
        centred = data.values - mean_last_dim_expanded
        data.values = centred
        return data
    
    @staticmethod
    def standardize(x):
        """Scaling variances to mean variance of the group."""
        return (x.data / x.data.std(dim=...)) * x.mean_std

    def zscoring(self, data: xr.DataArray) -> xr.DataArray:
        """Apply z-scoring based on configuration."""
        if not self.apply_zscore:
            return data
    
        match self.apply_zscore:
            case ApplyZScore.ALL:
                return (
                    data.stack(comp=["component"])
                    .groupby("comp", squeeze=False)
                    .map(zscore_xarray)
                    .unstack()
                )
            case ApplyZScore.PARTICIPANT:
                return (
                    data.stack(participant_comp=["participant", "component"])
                    .groupby("participant_comp", squeeze=False)
                    .map(zscore_xarray)
                    .unstack()
                )
            case ApplyZScore.TRIAL:
                return (
                    data.stack(trials=["participant", "epoch", "component"])
                    .groupby("trials", squeeze=False)
                    .map(zscore_xarray)
                    .unstack()
                )
        return data

    def data_format(
        self,
        data: xr.DataArray,
        weights: xr.DataArray,
        preprocessing_model: Any,
    ) -> xr.DataArray:
        """Finalize the transformation: transpose, reassign coords, stack, and store attributes."""
        data = data.transpose("participant", "epoch", "sample", "component")
        data.attrs["sfreq"] = self.sfreq
        data.attrs["offset"] = self.offset_after * self.sfreq
        self.data = data.stack(
            all_samples=["participant", "epoch", "sample"]).dropna(dim="all_samples")
        self.weights = weights
        self.preprocessing_model = preprocessing_model
        return self.data
        
class Standard(BasePreprocessing):
    """Transforms epoched data using a PCA for HMP analysis.

    Project channels to principal components space. The PCA is performed on subject-averaged covariance matrix among electrodes. The number of PC is either declared at initialization or a plot opens with a prompt to select based on the scree plot of the PCA.
    
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
    apply_standard : bool
        Whether to standardize variance across participants.
    apply_zscore : Union[bool, str, ApplyZScore]
        Z-scoring mode: 'all', 'participant', 'trial', or 'dont_apply'.
    centering : bool
        Whether to center the data across the last dimension.
    copy : bool
        Whether to copy the data before preprocessing.
    verbose : bool
        Whether to print rejection/cropping details.
    n_comp : int
        Number of components in the PCA to retain for projection. If None (default), an interactive prompt will open asking for selection of a number given the PCA screeplot
    """
    
    def __init__(
        self,
        epoch_data: xr.Dataset,
        interval_id: str = 'rt',
        offset_after: float = 0,
        too_short: Optional[float] = None,
        too_long: Optional[float] = None,
        reject_threshold: Optional[float] = None,
        apply_standard: bool = False,
        apply_zscore: Union[bool, str, ApplyZScore] = ApplyZScore.PARTICIPANT,
        centering: bool = False,
        copy: bool = False,
        verbose: bool = True,
        n_comp: Optional[int] = None,
    ):
        super().__init__(
            interval_id=interval_id,
            offset_after=offset_after,
            too_short=too_short,
            too_long=too_long,
            reject_threshold=reject_threshold,
            verbose=verbose,
            apply_standard=apply_standard,
            apply_zscore=apply_zscore,
            centering=centering,
            copy=copy,
        )
        self.n_comp = n_comp
        data = self.common_preprocess(epoch_data)
        indiv_data = np.zeros(
            (data.sizes["participant"], data.sizes["channel"], data.sizes["channel"])
        )
        for i in range(data.sizes["participant"]):
            x_i = np.squeeze(data.data[i])
            indiv_data[i] = np.mean(
                [
                    np.cov(x_i[trial, :, ~np.isnan(x_i[trial, 0, :])].T)
                    for trial in range(x_i.shape[0])
                    if ~np.isnan(x_i[trial, 0, :]).all()
                ],
                axis=0,
            )
        pca_ready_data = np.mean(np.array(indiv_data), axis=0)
        # Performing spatial PCA on the average var-cov matrix

        if self.n_comp is None:
            self.n_comp = self.user_input_n_comp(data=pca_ready_data)

        weights, preprocessing_model = self._pca(pca_ready_data, self.n_comp,
                                                 data.coords["channel"].values)
        data = data @ weights
        data = self.zscoring(data)
        self.data = self.data_format(
            data, weights, preprocessing_model
        )
        self.weights = weights
        self.preprocessing_model = preprocessing_model

    @staticmethod
    def user_input_n_comp(data):

        n_comp = np.shape(data)[0] - 1
        fig, ax = plt.subplots(1, 2, figsize=(0.2 * n_comp, 4))
        pca = PCA(n_components=n_comp, svd_solver="full", copy=False)  # selecting PCs
        pca.fit(data)

        ax[0].plot(np.arange(pca.n_components) + 1, pca.explained_variance_ratio_, ".-")
        ax[0].set_ylabel("Normalized explained variance")
        ax[0].set_xlabel("Component")
        ax[1].plot(np.arange(pca.n_components) + 1, np.cumsum(pca.explained_variance_ratio_), ".-")
        ax[1].set_ylabel("Cumulative normalized explained variance")
        ax[1].set_xlabel("Component")
        plt.tight_layout()
        plt.show()

        # TODO: needs user input validation?
        n_comp = int(
            input(
                f"How many PCs (95 and 99% explained variance at component "
                f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1} and "
                f"n{np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0] + 1}; "
                f"components till n{np.where(pca.explained_variance_ratio_ >= 0.01)[0][-1] + 1} "
                f"explain at least 1%)?"
            )
        )

        return n_comp

    @staticmethod
    def _pca(pca_ready_data: xr.DataArray, n_comp: int, channel) -> xr.DataArray:
        pca = PCA(n_components=n_comp, svd_solver="full")  # selecting Principale components (PC)
        pca.fit(pca_ready_data)
        # Rebuilding pca PCs as xarray to ease computation
        coords = dict(channel=("channel", channel), component=("component", np.arange(n_comp)))
        pca_weights = xr.DataArray(pca.components_.T, dims=("channel", "component"), coords=coords)
        return pca_weights, pca


class Identity(BasePreprocessing):
    """Transforms epoched data using a the Identity matrix for HMP analysis.

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
    apply_standard : bool
        Whether to standardize variance across participants.
    apply_zscore : Union[bool, str, ApplyZScore]
        Z-scoring mode: 'all', 'participant', 'trial', or 'dont_apply'.
    centering : bool
        Whether to center the data across the last dimension.
    copy : bool
        Whether to copy the data before preprocessing.
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
        apply_standard: bool = False,
        apply_zscore: Union[bool, str, ApplyZScore] = False,
        centering: bool = False,
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
            apply_standard=apply_standard,
            apply_zscore=apply_zscore,
            centering=centering,
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
        preprocessing_model = None

        # Final formatting
        data = self.zscoring(data)
        self.data = self.data_format(
            data, weights, preprocessing_model
        )
        self.weights = weights
        self.preprocessing_model = preprocessing_model
        
class Arbitrary(BasePreprocessing):
    """Transforms epoched data using a a custom linear combination for HMP analysis.

    Projects channels into a custom space given by a n_chan x n_component matrix. See xample matrix when applying `Standard` class.
    
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
    apply_standard : bool
        Whether to standardize variance across participants.
    apply_zscore : Union[bool, str, ApplyZScore]
        Z-scoring mode: 'all', 'participant', 'trial', or 'dont_apply'.
    centering : bool
        Whether to center the data across the last dimension.
    copy : bool
        Whether to copy the data before preprocessing.
    verbose : bool
        Whether to print rejection/cropping details.
    weights: xr.DataArray
        Custom linear combination of channels as an xarray.DataArray with 'channel' and 'component' dimensions and the weights
    """
    
    def __init__(
        self,
        epoch_data: xr.Dataset,
        weights: xr.DataArray,
        interval_id: str = 'rt',
        offset_after: float = 0,
        too_short: Optional[float] = None,
        too_long: Optional[float] = None,
        reject_threshold: Optional[float] = None,
        apply_standard: bool = False,
        apply_zscore: Union[bool, str, ApplyZScore] = ApplyZScore.PARTICIPANT,
        centering: bool = False,
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
            apply_standard=apply_standard,
            apply_zscore=apply_zscore,
            centering=centering,
            copy=copy,
        )
        # Preprocessing
        data = self.common_preprocess(epoch_data)

        # Projection
        data = data @ weights
        preprocessing_model = 'custom'

        # Final formatting
        data = self.zscoring(data)
        self.data = self.data_format(
            data, weights, preprocessing_model
        )
        self.weights = weights
        self.preprocessing_model = preprocessing_model


class MCCA_aligned(BasePreprocessing):
    """Transforms epoched data using MCCA for HMP analysis.

    Applies a PCA per subject and align the components across the PCAs using MCCA.
    
    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    n_comp : int, optional
        Number of components to retain in the final MCCA space.
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
    apply_standard : bool
        Whether to standardize variance across participants.
    apply_zscore : Union[bool, str, ApplyZScore]
        Z-scoring mode: 'all', 'participant', 'trial', or 'dont_apply'.
    centering : bool
        Whether to center the data across the last dimension.
    copy : bool
        Whether to copy the data before preprocessing.
    verbose : bool
        Whether to print rejection/cropping details.
    cov : bool, optional
        Whether to apply PCA/MCCA to the variance-covariance matrix (True)
        or the epoched data (False).
    averaged : bool, optional
        Whether to apply MCCA on the averaged ERP (True) or single-trial ERP (False).
        Only applicable for the MCCA method when cov=False. Default is False.
    n_ppcas : int, optional
        For the MCCA method, controls the number of components retained for by-participant PCAs.  If None (default), n_ppcas * 3 is choosen
        Default is None.
    mcca_reg : float, optional
        Regularization parameter for the MCCA computation. Default is 0.
    """
    
    def __init__(
        self,
        epoch_data: xr.Dataset,
        n_comp: int,
        interval_id: str = 'rt',
        offset_after: float = 0,
        too_short: Optional[float] = None,
        too_long: Optional[float] = None,
        reject_threshold: Optional[float] = None,
        apply_standard: bool = False,
        apply_zscore: Union[bool, str, ApplyZScore] = ApplyZScore.PARTICIPANT,
        centering: bool = True,
        copy: bool = False,
        verbose: bool = True,
        cov: bool = False,
        averaged: bool = True,
        n_ppcas: int = None,
        mcca_reg: float = 0
    ):
        super().__init__(
            interval_id=interval_id,
            offset_after=offset_after,
            too_short=too_short,
            too_long=too_long,
            reject_threshold=reject_threshold,
            verbose=verbose,
            apply_standard=apply_standard,
            apply_zscore=apply_zscore,
            centering=centering,
            copy=copy,
        )
        warn('The use of MCCA is experimental, not yet meant for inference')
        # Preprocessing
        data = self.common_preprocess(epoch_data)
        ori_coords = data.drop_vars("channel").coords

        # Projection
        if data.sizes["participant"] == 1:
            raise ValueError("MCCA cannot be applied to only one participant")

        if n_ppcas is None:
            n_ppcas = n_comp * 3
        mcca_m = mcca.MCCA(n_components_pca=n_ppcas, n_components_mcca=n_comp, r=mcca_reg)
        if cov:
            fitted_data = data.transpose("participant", "epoch", "sample", "channel").data
            ccs = mcca_m.obtain_mcca_cov(fitted_data)
        else:
            if averaged:
                fitted_data = (
                    data.mean("epoch").transpose("participant", "sample", "channel").data
                )
            else:
                fitted_data = (
                    data.stack({"all": ["epoch", "sample"]})
                    .transpose("participant", "all", "channel")
                    .data
                )
            ccs = mcca_m.obtain_mcca(fitted_data)
        trans_ccs = np.tile(
            np.nan,
            (data.sizes["participant"],
             data.sizes["epoch"],
             data.sizes["sample"],
             ccs.shape[-1]),
        )
        for i, part in enumerate(data.participant):
            trans_ccs[i] = mcca_m.transform_trials(
                data.sel(participant=part).transpose(
                    "epoch", "sample", "channel").data.copy()
            )
        data = xr.DataArray(
            trans_ccs,
            dims=["participant", "epoch", "sample", "component"],
            coords=dict(
                participant=data.participant,
                epoch=data.epoch,
                sample=data.sample,
                component=np.arange(n_comp),
            ),  # n_comp
        )
        data = data.assign_coords(ori_coords)
        weights = mcca_m.mcca_weights
        preprocessing_model = mcca_m

        # Final formatting
        data = self.zscoring(data)
        self.data = self.data_format(
            data, weights, preprocessing_model
        )
        self.weights = weights
        self.preprocessing_model = preprocessing_model
