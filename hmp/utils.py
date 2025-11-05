"""Functions to transform the input data and the estimates."""

from warnings import warn

import numpy as np
import xarray as xr
from pandas import MultiIndex

def reject_crop_epochs(epoch_data:xr.Dataset,
                         sfreq:float,
                         interval_id:str = 'interval',
                         offset_after:float = 0,
                         too_short:float=None,
                         too_long:float=None,
                         reject_threshold:float=None,
                         verbose:bool=True):
    """
    Crop each epoch from time 0 of the epoch to its interval with optional rejection criteria.

    For epoch in the `epoch_data` xr.Dataset, this function trims the epoch data from epoch
    time 0 (e.g. stimulus onset) up to the specified interval (e.g. response), optionally including a
    fixed offset after the interval.
    Epochs whose interval exceeds specified lower and upper limits are rejected, and additional rejection
    can be applied based on signal amplitude thresholds in the interval.

    Parameters
    ----------
    data_epoch : xr.Dataset
        Array of epoched EEG/MEG data, shaped (n_epochs, n_channels, n_samples) from io module.
        Should include the intervals in `interval_id`
    interval_id : str
        varialbe in the `Data variables` of the xr.Dataset to be used for interval crop
    too_short : float
        Minimum value of the intervals; epochs with shorter intervals are rejected.
    too_long : float
        Maximum value of the intervals; epochs with longer intervals are rejected.
    reject_threshold : float or None
        Maximum allowed signal amplitude for an epoch; epochs exceeding this are rejected.
    verbose : bool
        If True, print detailed processing steps.

    Returns
    -------
    epoch_data : np.ndarray
        Array of cropped epoch data that passed all criteria.
    """
    epoch_data = epoch_data.sel(sample=range(0, int(epoch_data.sample.max())+1))
    epoch_data = epoch_data.stack(trial=["participant", "epoch"])
    epoch_data = epoch_data.transpose('trial','channel','sample')
    rts_arr = epoch_data.coords[interval_id].values.copy()
    if np.nanmean(rts_arr) > 100:
        warn(f"Found intervals with an average value of {np.round(np.nanmean(rts_arr),2)}"
            ", assuming intervals are in milliseconds and converting to seconds")
        rts_arr /= 1000
    offset_after_samples = int(np.rint(offset_after * sfreq))

    if too_long is None:
        too_long = float(epoch_data.sample.max()) / sfreq
    if too_short is None:
        too_short = 1 / sfreq
    if too_long < 0 or too_short < 0:
        raise ValueError("Limit to intervals cannot be negative")

    rts_arr[rts_arr > too_long] = 0  # removes intervals above x sec
    rts_arr[rts_arr < too_short] = 0  # removes intervals below x sec, determines max events
    rt_criteria_rej = len(rts_arr[rts_arr == 0]) 
    inexistant_rej = len(rts_arr[np.isnan(rts_arr)])
    rts_arr[np.isnan(rts_arr)] = 0  # rejected during epoching or inexistant
    # Converting to samples
    rts_arr = np.rint(rts_arr * sfreq).astype(int)
    
    epoch_data = epoch_data.sel(sample=range(0, int(rts_arr.max())+1+offset_after_samples))
    
    assert len(rts_arr[rts_arr > 0]) > 0, "No intervals are between the requested limits of "\
        f"minimum {too_short} and maximum {too_long} seconds"

    min_rt = min(rts_arr[rts_arr > 0])
    if min_rt < 10:
        warn(f"The shortest interval is less than 10 samples. "
             "Consider specifying too short trials using the `too_short` parameter "
             "or increasing sampling frequency of the signal.")

    if verbose:
        print(f"{len(rts_arr[rts_arr > 0])} intervals between {too_short} and "\
            f"{too_long} seconds.")
    cropped_data_epoch = np.empty(
        [
            len(epoch_data),
            epoch_data.sizes['channel'],
            max(rts_arr) + offset_after_samples,
        ], dtype=np.float32
    )
    cropped_data_epoch[:] = np.nan
    cropped_trigger = []
    trial_coord = []
    j = 0
    if reject_threshold is None:
        reject_threshold = np.inf
    rej = 0
    time0 = np.argmin(np.abs(epoch_data.sample.values))
    for i in range(len(epoch_data.data)):
        if rts_arr[i] > 0:
            # Crops the epochs to time 0 (stim onset) up to RT
            if (
                np.abs(epoch_data.values[i, :, time0 : time0 + rts_arr[i] + offset_after_samples])
                < reject_threshold
            ).all():
                cropped_data_epoch[j, :, : rts_arr[i] + offset_after_samples] = epoch_data.values[
                    i, :, time0 : time0 + rts_arr[i] + offset_after_samples
                ]
                j += 1
                trial_coord.append(epoch_data.trial[i].values)
            elif ~np.isnan(epoch_data.values[i, 0, time0]):
                rej += 1
            else: # assumes rejected before
                inexistant_rej += 1
    assert rej < len(cropped_data_epoch), 'All trials rejected, inspect intervals and rejection criterion'
    if verbose:
        print(f"Rejection summary: \n {rej} trials rejected based on threshold of {reject_threshold}"
         f"\n {rt_criteria_rej} trials rejected based on interval limit of {too_short, too_long}"
         f"\n {inexistant_rej} trials detected as inexisting (e.g. preprocessing prior to HMP) ")

    while np.isnan(cropped_data_epoch[-1]).all():  # Remove excluded epochs based on rejection
        cropped_data_epoch = cropped_data_epoch[:-1]

    cropped_data_epoch = xr.DataArray(
        data=cropped_data_epoch,
        dims=("trial", "channel", "sample"),
        attrs={
              "offset":offset_after_samples}
    )
    trial_coord = [tuple(arr.tolist()) for arr in trial_coord]
    updated_coords = dict(epoch_data.sel(trial=trial_coord).drop_vars('sample').coords)
    cropped_data_epoch = cropped_data_epoch.assign_coords(updated_coords)

    return cropped_data_epoch.unstack()

def zscore_xarray(data: xr.DataArray) -> xr.DataArray:
        """Zscore of the data in an xarray, avoiding any nans."""
        non_nan_mask = ~np.isnan(data.values)
        if non_nan_mask.any():  # if not everything is nan, calc zscore
            data.values[non_nan_mask] = (
                data.values[non_nan_mask] - data.values[non_nan_mask].mean()
            ) / data.values[non_nan_mask].std()
        return data

def stack_data(data):
    """Stack the data.

    Going from format [participant * epochs * sample * channel] to
    [sample * channel] with sample indexes starts and ends to delimitate the epochs.


    Parameters
    ----------
    data : xarray
        unstacked xarray data from transform_data() or anyother source yielding an xarray with
        dimensions [participant * epochs * sample * channel]
    subjects_variable : str
        name of the dimension for subjects ID

    Returns
    -------
    data : xarray.Dataset
        xarray dataset [sample * channel]
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)) and "component" not in data.dims:
        data = data.rename_dims({"channel": "component"})
    if "participant" not in data.dims:
        data = data.expand_dims("participant")
    data = data.stack(all_samples=["participant", "epoch", "sample"]).dropna(dim="all_samples")
    return data


def event_times(  # noqa: PLR0912
    estimates,
    duration=False,
    mean=False,
    add_rt=False,
    as_time=False,
    errorbars=None,
    estimate_method="max",
    add_stim=False,
    remove_offset=False,
):
    """Compute the likeliest peak times for each event.

    Parameters
    ----------
    estimates : xr.Dataset
        Estimated instance of an HMP model
    duration : bool
        Whether to compute peak location (False) or inter-peak duration (True)
    mean : bool
        Whether to compute the mean (True) or return the single trial estimates
        Note that mean and errorbars cannot both be true.
    add_rt : bool
        whether to append the last stage up to the RT
    as_time : bool
        if true, return time (ms) instead of sample
    errorbars : str
        calculate 95% confidence interval ('ci'), standard deviation ('std'),
        standard error ('se') on the times or durations, or None.
        Note that mean and errorbars cannot both be true.
    estimate_method : string
        'max' or 'mean', either take the max probability of each event on each trial, or the
        weighted average.
    add_stim: bool
        Adding stimulus as the first event (True) or let the first estimated HMP event be the
        first one (False, default)
    remove_offset: bool
        Whether to remove the eventual offset added to the reaction time

    Returns
    -------
    times : xr.DataArray
        Transition event peak or stage duration with trial*event dimensions or
        only event dimension if mean = True contains nans for missing stages.
    """
    assert not (mean and errorbars is not None), "Only one of mean and errorbars can be set."
    tstep = 1000 / estimates.sfreq if as_time else 1

    if estimate_method is None:
        estimate_method = "max"
    event_shift = 0
    eventprobs = estimates.fillna(0).copy()
    if estimate_method == "max":
        times = eventprobs.argmax("sample") - event_shift  # Most likely event location
    else:
        times = xr.dot(eventprobs, eventprobs.sample, dims="sample") - event_shift
    times = times.astype("float32")  # needed for eventual addition of NANs
    times_group = (
        times.groupby("group").mean("trial").values
    )  # take average to make sure it's not just 0 on the trial-group
    for c, e in np.argwhere(times_group == -event_shift):
        times[times["group"] == c, e] = np.nan

    if add_rt:
        rts = estimates.cumsum('sample').argmax('sample').max('event')+1
        if remove_offset:
            rts = rts-estimates.offset
        rts = xr.DataArray(rts)
        rts = rts.assign_coords(event=int(times.event.max().values + 1))
        rts = rts.expand_dims(dim="event")
        times = xr.concat([times, rts], dim="event")

    times = times * tstep
    if duration:  # taking into account missing events, hence the ugly code
        added = xr.DataArray(
            np.repeat(0, len(times.trial))[np.newaxis, :],
            coords={"event": [0], "trial": times.trial},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)
        for c in np.unique(times["group"].values):
            tmp = times.isel(trial=estimates["group"] == c).values
            # identify nan columns == missing events
            missing_evts = np.where(np.isnan(np.mean(tmp, axis=0)))[0]
            tmp = np.diff(
                np.delete(tmp, missing_evts, axis=1)
            )  # remove 0 columns, calc difference
            # insert nan columns (to maintain shape),
            for missing in missing_evts:
                tmp = np.insert(tmp, missing - 1, np.nan, axis=1)
            # add extra column to match shape
            tmp = np.hstack((tmp, np.tile(np.nan, (tmp.shape[0], 1))))
            times[estimates["group"] == c, :] = tmp
        times = times[:, :-1]  # remove extra column
    elif add_stim:
        added = xr.DataArray(
            np.repeat(0, len(times.trial))[np.newaxis, :],
            coords={"event": [0], "trial": times.trial},
        )
        times = times.assign_coords(event=times.event + 1)
        times = times.combine_first(added)

    if mean:
        times = times.groupby("group").mean("trial")
    elif errorbars:
        errorbars_model = np.zeros((len(np.unique(times["group"])), 2, times.shape[1]))
        if errorbars == "std":
            std_errs = times.groupby("group").reduce(np.std, dim="trial").values
            for c in np.unique(times["group"]):
                errorbars_model[c, :, :] = np.tile(std_errs[c, :], (2, 1))
        else:
            raise ValueError(
                "Unknown error bars, 'std' is for now the only accepted argument in the "
                "multigroup models"
            )
        times = errorbars_model
    return times


def event_channels(
    epoch_data,
    estimated,
    mean=True,
    peak=True,
    estimate_method="max",
    template=None,
):
    """Compute topographies for each trial.

    Parameters
    ----------
        epoch_data: xr.Dataset
            Epoched data
        estimated: xr.Dataset
            estimated model parameters and event probabilities
        mean: bool
            if True mean will be computed instead of single-trial channel activities
        peak : bool
            if true, return topography at peak of the event. If false, return topographies weighted
            by a normalized template.
        estimate_method : string
            'max' or 'mean', either take the max probability of each event on each trial, or the
            weighted average.
        template: int
            Length of the pattern in sample (e.g. 5 for a pattern of 50 ms with a 100Hz sampling
            frequency)

    Returns
    -------
        event_values: xr.DataArray
            array containing the values of each electrode at the most likely transition time
            contains nans for missing events
    """
    if estimate_method is None:
        estimate_method = "max"
    epoch_data = (
        epoch_data.stack(trial=["participant", "epoch"])
        .data
        .drop_duplicates("trial")
    )

    common_trial = np.intersect1d(
        estimated["trial"].values, epoch_data["trial"].values
    )
    epoch_data = epoch_data.sel(trial=common_trial, sample=estimated.sample)
    estimated = estimated.sel(trial=common_trial)
    n_events = estimated.event.count().values
    n_trial = estimated.trial.count().values
    n_channel = epoch_data.channel.count().values

    if not peak:
        normed_template = template / np.sum(template)

    times = event_times(estimated, mean=False, estimate_method=estimate_method,)
    times = times.sel(trial=common_trial)
    event_values = np.zeros((n_channel, n_trial, n_events))*np.nan
    for ev in range(n_events):
        for tr in range(n_trial):
            # If time is nan, means that no event was estimated for that trial/group
            if np.isfinite(times.values[tr, ev]):
                samp = int(times.values[tr, ev])
                if peak:
                    event_values[:, tr, ev] = epoch_data.values[:, samp, tr]
                else:
                    vals = epoch_data.values[:, samp : samp + template // 2, tr]
                    event_values[:, tr, ev] = np.dot(vals, normed_template[: vals.shape[1]])

    event_values = xr.DataArray(
        event_values,
        dims=[
            "channel",
            "trial",
            "event",
        ],
        coords={
            "trial": estimated.trial,
            "event": estimated.event,
            "channel": epoch_data.channel,
        },
    )

    event_values = event_values.assign_coords(
        group=("trial", times.group.data)
    )

    if mean:
        event_values = event_values.groupby("group").mean("trial")
    return event_values


def centered_activity(
    data,
    times,
    channel,
    event,
    n_samples=None,
    cut_after_event=0,
    baseline=0,
    cut_before_event=0,
    event_width=0,
):
    """Parse the single trial signal of channel in a given number of sample around one event.

    Parameters
    ----------
    data : xr.Dataset
        HMP data (untransformed but with trial and participant stacked)
    times : xr.DataArray
        Onset times as computed using onset_times()
    channel : list
        channel to pick for the parsing of the signal, must be a list even if only one
    event : int
        Which event is used to parse the signal
    n_samples : int
        How many sample to record after the event (default = maximum duration between event and
        the consecutive event)
    cut_after_event: int
        Which event after ```event``` to cut sample off, if 1 (Default) cut at the next event
    baseline: int
        How much sample should be kept before the event
    cut_before_event: int
        At which previous event to cut sample from, ```baseline``` if 0 (Default), no effect if
        baseline = 0
    event_width: int
        Duration of the fitted events, used when cut_before_event is True

    Returns
    -------
    centered_data : xr.Dataset
        Xarray dataset with electrode value (data) and trial event time (time) and with
        trial * sample dimension
    """
    if event == 0:  # no sample before stim onset
        baseline = 0
    elif event == 1:  # no event at stim onset
        event_width = 0
    if cut_before_event == 0:  # avoids searching before stim onset
        cut_before_event = event

    if n_samples is None:
        if cut_after_event is None:
            raise ValueError(
                "One of ```n_samples``` or ```cut_after_event``` has to be filled to use an upper"
                "limit"
            )
        n_samples = (
            max(times.sel(event=event + cut_after_event).data - times.sel(event=event).data) + 1
        )

    n_samples = np.rint(n_samples)
    baseline = np.rint(baseline)

    if 'epoch' in data.dims:
        data = (
            data.stack({'trial':['participant','epoch']})
            .data
            .drop_duplicates("trial")
        )
    common_trial = np.intersect1d(
        times["trial"].values, data["trial"].values
    )
    data = data.sel(trial=common_trial)
    times = times.sel(trial=common_trial)

    assert ~np.any(times > data.sample.max()),\
        "At least one trial is longer than the maximum possible sample.\
        Provided times should be in sample not on the millisecond scale"

    centered_data = np.tile(
        np.nan,
        (len(common_trial), len(channel), int(round(n_samples - baseline + 1))),
    )

    trial_times = np.zeros(len(common_trial)) * np.nan
    participants = []
    epochs = np.zeros(len(common_trial))
    for i, (trial, trial_dat) in enumerate(data.groupby("trial", squeeze=False)):
        participants.append(trial[0])
        epochs[i] = trial[1]
        if cut_before_event > 0:
            # Lower lim is baseline or the last sample of the previous event
            lower_lim = np.max(
                [
                    -np.max(
                        [
                            times.sel(event=event, trial=trial)
                            - times.sel(
                                event=event - cut_before_event, trial=trial
                            )
                            - event_width // 2,
                            0,
                        ]
                    ),
                    baseline,
                ]
            )
        else:
            lower_lim = 0
        if cut_after_event > 0:
            upper_lim = np.max(
                [
                    np.min(
                        [
                            times.sel(event=event + cut_after_event, trial=trial)
                            - times.sel(event=event, trial=trial)
                            - event_width // 2,
                            n_samples,
                        ]
                    ),
                    0,
                ]
            )
        else:
            upper_lim = n_samples

        # Determine sample in the signal to store
        start_idx = int(times.sel(event=event, trial=trial) + lower_lim)
        end_idx = int(times.sel(event=event, trial=trial) + upper_lim)
        trial_elec = trial_dat.sel(channel=channel, sample=slice(start_idx, end_idx))\
            .squeeze("trial")
        # If center, adjust to always center on the same sample if lower_lim < baseline
        baseline_adjusted_start = int(abs(baseline - lower_lim))
        baseline_adjusted_end = baseline_adjusted_start + trial_elec.shape[-1]
        trial_time_arr = slice(baseline_adjusted_start, baseline_adjusted_end)

        centered_data[i, :, trial_time_arr] = trial_elec
        trial_times[i] = times.sel(event=event, trial=trial)

    trial_x_part = xr.Coordinates.from_pandas_multiindex(
        MultiIndex.from_arrays([participants, epochs], names=("participant", "epoch")),
        "trial",
    )
    centered_data = xr.Dataset(
        {
            "data": (("trial", "channel", "sample"), centered_data),
            "times": (("trial"), trial_times),
        },
        {"channel": channel, "sample": np.arange(centered_data.shape[-1]) + baseline},
        attrs={"event": event},
    )

    return centered_data.assign_coords(trial_x_part)


def condition_selection(preprocessed_data, condition_string, variable="event", method="equal"):
    """Select a subset from preprocessed_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    preprocessed_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in preprocessed_data that is used for condition selection
    method : str
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of preprocessed_data.
    """
    unstacked = preprocessed_data.unstack()
    unstacked[variable] = unstacked[variable].fillna("")
    if method == "equal":
        unstacked = unstacked.where(unstacked[variable] == condition_string, drop=True)
        stacked = stack_data(unstacked)
    elif method == "contains":
        unstacked = unstacked.where(unstacked[variable].str.contains(condition_string), drop=True)
        stacked = stack_data(unstacked)
    else:
        warn("unknown method, returning original data")
        stacked = preprocessed_data
    return stacked


def condition_selection_epoch(epoch_data, condition_string, variable="event", method="equal"):
    """Select a subset from epoch_data.

    The function selects epochs for which 'condition_string' is in 'variable' based on 'method'.

    Parameters
    ----------
    epoch_data : xr.Dataset
        transformed EEG data for hmp, e.g. from utils.read_mne_data()
    condition_string : str | num
        condition indicator for selection
    variable : str
        variable present in preprocessed_data that is used for condition selection
    method : str
        'equal' selects equal trial, 'contains' selects trial in which conditions_string
        appears in variable

    Returns
    -------
    data : xr.Dataset
        Subset of preprocessed_data.
    """
    if len(epoch_data.dims) == 4:
        stacked_epoch_data = epoch_data.stack(trial=("participant", "epoch")).dropna(
            "trial", how="all"
        )

    if method == "equal":
        stacked_epoch_data = stacked_epoch_data.where(
            stacked_epoch_data[variable] == condition_string, drop=True
        )
    elif method == "contains":
        stacked_epoch_data = stacked_epoch_data.where(
            stacked_epoch_data[variable].str.contains(condition_string), drop=True
        )
    return stacked_epoch_data.unstack()


def participant_selection(preprocessed_data, participant):
    """Select a participant from preprocessed_data.

    Parameters
    ----------
    preprocessed_data : xr.Dataset
        transformed EEG data for hmp, from utils.transform_data
    participant : str | num
        Name of the participant

    Returns
    -------
    data : xr.Dataset
        Subset of preprocessed_data.
    """
    unstacked = preprocessed_data.unstack().sel(participant=participant)
    stacked = stack_data(unstacked)
    return stacked
