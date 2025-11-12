"""Transforms epoched data using a PCA for HMP analysis.

Project channels to principal components space.
The PCA is performed on subject-averaged covariance matrix among electrodes.
The number of PC is either declared at initialization or a plot opens with
 a prompt to select based on the scree plot of the PCA.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.linalg import eigh

from hmp.transformers.base import BaseTransformer


class ProjPCA(BaseTransformer):
    """Transforms epoched data using a PCA for HMP analysis.

    Project channels to principal components space.
    The PCA is performed on subject-averaged covariance matrix among electrodes.
    The number of PC is either declared at initialization or a plot opens with
     a prompt to select based on the scree plot of the PCA.

    Parameters
    ----------
    epoch_data : xr.Dataset
        Input EEG data with dimensions [participant, epoch, sample, channel], from `io` module
    interval_id: str
        Name of the variable that contains the trial intervals in the epoch_data used for cropping.
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
        Return the components with unit-variance
    center : bool
        Whether to center the data across the last dimension before projection
    copy : bool
        Whether to copy the data before preprocessing.
    verbose : bool
        Whether to print rejection/cropping details.
    n_comp : int
        Number of components in the PCA to retain for projection.
        If None (default), a prompt will open asking to select a number
    """

    def __init__(#noqa: PLR0913
        self,
        epoch_data: xr.Dataset,
        interval_id: str = 'rt',
        offset_after_end: float = 0,
        min_duration: float = 0,
        max_duration: float =float('Inf'),
        reject_threshold: Optional[float] = None,
        common_variance: bool = False,
        whiten: bool = True,
        center: bool = True,
        copy: bool = False,
        verbose: bool = True,
        n_comp: Optional[int] = None,
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

        self.n_comp = n_comp
        data = self.common_preprocess(epoch_data)

        # Performing spatial PCA on the average var-cov matrix
        pca_ready_data = np.zeros((data.sizes["channel"], data.sizes["channel"]), dtype=data.dtype)
        count = 0
        for i in range(data.sizes["trial"]):
            x_i = np.squeeze(data.isel(trial=i).values)
            x_i = x_i[:, ~np.isnan(x_i[0, :])]
            if not self.center: #ensure cov computation
                x_i -= np.mean(x_i)
            cov_i = x_i @ x_i.T
            # Regularization
            cov_i += 1e-15 * np.eye(data.sizes["channel"])
            pca_ready_data += cov_i
            count += 1
        pca_ready_data /= count
        pca_ready_data
        if self.n_comp is None:
            self.n_comp = self.user_input_n_comp(pca_ready_data,
                                                 data.sizes["channel"],
                                                 data.coords["channel"].values)

        weights, _ = self._pca(pca_ready_data, self.n_comp,
                            data.coords["channel"].values)
        data = data @ weights
        self.data = self.data_format(
            data, weights
        )

    def _pca(self,
             pca_ready_data: xr.DataArray,
             n_comp: int,
             channel: xr.DataArray) -> xr.DataArray:
        # Mostly from https://github.com/coffeine-labs/coffeine/blob/main/coffeine/spatial_filters.py
        eigvals, eigvecs = eigh(pca_ready_data)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :n_comp]
        # Rebuilding as xarray to ease computation
        coords = dict(channel=("channel", channel), component=("component", np.arange(n_comp)))
        pca_weights = xr.DataArray(evecs, dims=("channel", "component"), coords=coords)
        return pca_weights, eigvals[ix]

    def user_input_n_comp(self,
                          data: xr.DataArray,
                          n_comp: int,
                          channel: xr.DataArray) -> int:

        n_comp = np.shape(data)[0] - 1
        fig, ax = plt.subplots(1, 2, figsize=(0.2 * n_comp, 4))
        pca, eigenvalues = self._pca(data, n_comp, channel)
        explained_variance_ratio = eigenvalues/np.sum(eigenvalues)
        ax[0].plot(explained_variance_ratio, ".-")
        ax[0].set_ylabel("Normalized explained variance")
        ax[0].set_xlabel("Component")
        ax[1].plot(np.cumsum(explained_variance_ratio), ".-")
        ax[1].set_ylabel("Cumulative normalized explained variance")
        ax[1].set_xlabel("Component")
        plt.tight_layout()
        plt.show()

        # TODO: needs user input validation?
        n_comp = int(
            input(
                f"How many PCs (95 and 99% explained variance at component "
                f"n{np.where(np.cumsum(explained_variance_ratio) >= 0.95)[0][0] + 1} and "
                f"n{np.where(np.cumsum(explained_variance_ratio) >= 0.99)[0][0] + 1}; "
                f"components till n{np.where(explained_variance_ratio >= 0.01)[0][-1] + 1} "
                f"explain at least 1%)?"
            )
        )

        return n_comp
