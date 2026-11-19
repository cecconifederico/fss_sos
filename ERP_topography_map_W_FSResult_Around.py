# -*- coding: utf-8 -*-
"""ERP spatial maps A/W and continuous FSResult from BrainVision EEG.

Spyder script with independent ERP A/W topographies.
Run all (F5) or execute sections separately.
Place BrainVision files alongside the script or change eeg_file.
"""



# %% Imports
# This cell imports the Python libraries required for the entire EEG analysis workflow.

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% Function definitions
# This function opens a BrainVision EEG recording with data loaded into memory for subsequent processing.

def read_prep_eeg(eeg_file_path_vhdr, low_freq=None, high_freq=None):
    """Load BrainVision data, classify auxiliary channels and retain EEG channels."""
    raw = mne.io.read_raw_brainvision(eeg_file_path_vhdr, preload=True)
    auxiliary_types = {"Mano DX": "misc", "Mano SX": "misc", "ECG": "ecg", "pressione": "misc"}
    raw.set_channel_types({name: kind for name, kind in auxiliary_types.items() if name in raw.ch_names})
    if low_freq is not None or high_freq is not None:
        raw = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
    return raw.pick("eeg")

# This function builds the event related potential from the EEG recording by aligning data segments with annotated events.

def get_erp(raw, epoch_pre_ms, epoch_post_ms, baseline_start_ms,
            baseline_end_ms, min_trigger_distance):
    """Average event-aligned EEG epochs and remove the original post-trigger baseline."""
    sfreq = float(raw.info["sfreq"])
    pre = int((epoch_pre_ms / 1000.0) * sfreq)
    post = int((epoch_post_ms / 1000.0) * sfreq)
    if pre < 0 or post <= 0:
        raise ValueError("Epoch durations must satisfy pre >= 0 and post > 0.")

    events, _ = mne.events_from_annotations(raw)
    if len(events) == 0:
        raise ValueError("No annotated events found in the EEG recording.")
    samples = events[:, 0]
    clean = [samples[0]]
    for sample in samples[1:]:
        if sample - clean[-1] > min_trigger_distance:
            clean.append(sample)

    epochs = []
    for trigger in clean:
        if trigger - pre < 0 or trigger + post > raw.n_times:
            continue
        epoch, _ = raw[:, trigger - pre:trigger + post]
        epochs.append(epoch)
    if not epochs:
        raise ValueError("No complete epochs remain after trigger filtering.")

    erp = np.mean(np.stack(epochs, axis=0), axis=0)
    b_start = pre + int((baseline_start_ms / 1000.0) * sfreq)
    b_end = pre + int((baseline_end_ms / 1000.0) * sfreq)
    if not (0 <= b_start < b_end <= erp.shape[1]):
        raise ValueError("Baseline interval must lie inside the ERP epoch.")
    baseline = np.mean(erp[:, b_start:b_end], axis=1, keepdims=True)
    time_epoch = np.arange(-pre, post) / sfreq
    return erp - baseline, time_epoch

# This function constructs the spatial ERP vector A without plotting any figure.

def compute_A(erp_mean, time_epoch, erp_time, erp_around=0.0, normalize=True):
    """Select the time neighborhood, average each electrode, and optionally normalize."""
    if not (np.isfinite(erp_around) and 0.0 <= erp_around <= 5.0):
        raise ValueError("erp_around must be between 0 and 5 milliseconds, inclusive.")
    if not np.isfinite(erp_time):
        raise ValueError("erp_time must be a finite number of milliseconds.")
    time_epoch = np.asarray(time_epoch, dtype=float)
    erp_mean = np.asarray(erp_mean, dtype=float)
    if time_epoch.ndim != 1 or time_epoch.size == 0:
        raise ValueError("time_epoch must contain at least one time sample.")
    if erp_mean.ndim != 2 or erp_mean.shape[1] != time_epoch.size:
        raise ValueError("ERP columns and time_epoch samples must match.")
    target_s = erp_time / 1000.0
    if target_s < time_epoch[0] or target_s > time_epoch[-1]:
        raise ValueError(f"erp_time={erp_time} ms is outside the epoch time range.")
    closest_index = int(np.argmin(np.abs(time_epoch - target_s)))
    actual_time_ms = float(time_epoch[closest_index] * 1000.0)
    if erp_around == 0:
        selected_indices = np.array([closest_index], dtype=int)
    else:
        radius_s = erp_around / 1000.0
        # Include both boundaries when they coincide with sampled times.
        tolerance_s = max(1e-12, np.finfo(float).eps * max(1.0, abs(target_s)) * 10)
        selected_indices = np.flatnonzero(
            (time_epoch >= target_s - radius_s - tolerance_s) &
            (time_epoch <= target_s + radius_s + tolerance_s)
        )
        if not selected_indices.size:
            raise ValueError("No recorded ERP samples fall inside the requested time window.")
    # Average across selected time samples, retaining one mean per EEG channel.
    potential = erp_mean[:, selected_indices].mean(axis=1)
    if normalize:
        denominator = potential.sum()
        if np.isclose(denominator, 0.0, atol=1e-15, rtol=0):
            raise ValueError("Cannot normalize A: ERP electrode values sum to (nearly) zero.")
        A = potential / denominator
    else:
        A = potential.copy()
    return A, potential, actual_time_ms, selected_indices

# This function converts the previously computed spatial ERP vector A into its Moore Penrose pseudoinverse W.

def compute_W(A):
    """Compute Moore-Penrose pseudoinverse of a single topographic row vector."""
    A = np.asarray(A, dtype=float).reshape(-1)
    if not np.any(A):
        raise ValueError("Cannot compute useful W from an all-zero A vector.")
    return np.linalg.pinv(A.reshape(1, -1)).reshape(-1)

# This function plots one scalp topography from a supplied vector of electrode values, without recomputing ERP data, A, or W.

def plot_topography(raw, values, time_ms, title_prefix="A", cmap="jet",
                    save_path=None, dpi=300, view_min=None, view_max=None):
    """Plot exactly one 2D scalp map; do not calculate or normalize its values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    info = raw.info.copy()
    picks = mne.pick_types(info, eeg=True)
    info = mne.pick_info(info, picks)
    if values.size != len(info.ch_names):
        raise ValueError(f"Map has {values.size} values but EEG has {len(info.ch_names)} channels.")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    if view_min is not None and view_max is not None and view_min >= view_max:
        raise ValueError("view_min must be smaller than view_max.")
    fig, ax = plt.subplots(figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=cmap,
        vlim=(view_min, view_max)
    )
    ax.set_title(f"{title_prefix} - {time_ms:.1f} ms")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax

# %% Analysis configuration
# This cell centralizes run parameters so the analysis can be configured without editing function bodies.

eeg_file = str(Path(__file__).resolve().parent / "Smits11.vhdr")
low_freq = 1.0
high_freq = None

# Recording frequency: populated from raw.info after loading, never hard-coded
sfreq = None

# Epoch and baseline windows, in milliseconds relative to the event
epoch_pre_ms = 20
epoch_post_ms = 100
baseline_start_ms = 11
baseline_end_ms = 14
min_trigger_distance = 100  # samples; preserve original strict > comparison

# ERP centers, in milliseconds; one independent A and W per center
erp_numerosity = 2
erp_values = [20, 23]
erp_around = 1.0         # milliseconds, allowed range [0, 5]
normalize_erp = True
FS_flag = False           # True: compute and plot one FSResult per W

# Figure settings
cmap = "jet"
view_scale_target = "A"   # "A" or "W": which group uses manual view_min/view_max
view_min = None           # manual lower limit for selected group; None = automatic
view_max = None           # manual upper limit for selected group; None = automatic
save_figures = True

# %% Read EEG
# This cell loads the configured BrainVision recording and applies filtering settings.

raw = read_prep_eeg(eeg_file, low_freq=low_freq, high_freq=high_freq)
sfreq = float(raw.info["sfreq"])
print(f"EEG sampling frequency: {sfreq:g} Hz")
print(f"EEG channels: {len(raw.ch_names)}")

# %% Compute ERP
# This cell computes the trial averaged ERP waveform using the prepared EEG recording and the centrally defined epoch, baseline, and trigger settings.

erp_mean, time_epoch = get_erp(
    raw, epoch_pre_ms, epoch_post_ms, baseline_start_ms,
    baseline_end_ms, min_trigger_distance
)
print("ERP matrix shape:", erp_mean.shape)
print("ERP time range (ms):", time_epoch[0] * 1000, "to", time_epoch[-1] * 1000)

# %% Plot ERP waveforms
# This cell visualizes the complete baseline corrected ERP time course for each retained EEG channel.

for i in range(erp_mean.shape[0]):
    plt.plot(time_epoch, erp_mean[i])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.show()

# %% Compute A and W for all ERP centers
# Validate the requested number of ERP centers and calculate each A/W pair independently.

if isinstance(erp_numerosity, bool) or not isinstance(erp_numerosity, (int, np.integer)) or erp_numerosity < 1:
    raise ValueError("erp_numerosity must be a positive integer.")
if len(erp_values) != erp_numerosity:
    raise ValueError("erp_values must contain exactly erp_numerosity entries.")
if not isinstance(FS_flag, (bool, np.bool_)):
    raise TypeError("FS_flag must be a boolean (True or False).")

results = []  # ordered list of per-center dictionaries, including optional FSResult
for idx, erp_time in enumerate(erp_values, start=1):
    A, A_raw, actual_erp_time, selected_erp_indices = compute_A(
        erp_mean, time_epoch, erp_time, erp_around=erp_around,
        normalize=normalize_erp
    )
    W = compute_W(A)
    results.append({
        "erp_time": erp_time,
        "actual_erp_time": actual_erp_time,
        "selected_erp_indices": selected_erp_indices,
        "A": A,
        "A_raw": A_raw,
        "W": W,
        "FSResult": None,
    })
    print(f"\nERP {idx}/{erp_numerosity}: center {erp_time:g} ms "
          f"(nearest sampled {actual_erp_time:.3f} ms)")
    print(f"Window: [{erp_time-erp_around:.3f}, "
          f"{erp_time+erp_around:.3f}] ms; "
          f"samples averaged: {len(selected_erp_indices)}")
    print("A shape:", A.shape, "W shape:", W.shape)
    print("A:", A)
    print("W:", W)

# %% Color bounds for A and W
# A and W have independent group-wide ranges. Manual limits apply only to
# view_scale_target; the other group always uses its automatic common range.

if view_scale_target not in ("A", "W"):
    raise ValueError('view_scale_target must be "A" or "W".')
if view_min is not None and not np.isfinite(view_min):
    raise ValueError("view_min must be finite or None.")
if view_max is not None and not np.isfinite(view_max):
    raise ValueError("view_max must be finite or None.")

color_bounds = {}
for group in ("A", "W"):
    group_values = np.concatenate([item[group] for item in results])
    if not np.all(np.isfinite(group_values)):
        raise ValueError(f"All {group} coefficients must be finite.")
    group_min = float(np.min(group_values))
    group_max = float(np.max(group_values))
    if group == view_scale_target:
        group_min = group_min if view_min is None else float(view_min)
        group_max = group_max if view_max is None else float(view_max)
    if group_min >= group_max:
        # Constant maps require a nonzero color interval for plotting.
        if group_min == group_max and (group != view_scale_target or
                                       (view_min is None and view_max is None)):
            padding = max(abs(group_min) * 0.01, 1e-12)
            group_min -= padding
            group_max += padding
        else:
            raise ValueError(f"Invalid {group} color range: min must be < max.")
    color_bounds[group] = (group_min, group_max)
    print(f"Common {group} color scale: [{group_min:.6g}, {group_max:.6g}]")

# %% Plot the individual A and W topographies
# Display one independent A map and one independent W map for each requested ERP center.

for idx, item in enumerate(results, start=1):
    center = item["erp_time"]
    suffix = f"{idx:02d}_ERP_{center:g}ms"
    item["fig_A"], item["ax_A"] = plot_topography(
        raw, item["A"], item["actual_erp_time"],
        title_prefix=f"A #{idx} (±{erp_around:g} ms)", cmap=cmap,
        save_path=f"A_{suffix}.png" if save_figures else None,
        view_min=color_bounds["A"][0], view_max=color_bounds["A"][1],
    )
    item["fig_W"], item["ax_W"] = plot_topography(
        raw, item["W"], item["actual_erp_time"],
        title_prefix=f"W #{idx} (±{erp_around:g} ms)", cmap=cmap,
        save_path=f"W_{suffix}.png" if save_figures else None,
        view_min=color_bounds["W"][0], view_max=color_bounds["W"][1],
    )

# %% Optional FSResult computation and plots
# FSResult is generated only when FS_flag is True.

if FS_flag:
    eeg_data = raw.get_data()
    time = np.arange(eeg_data.shape[1]) / sfreq
    for idx, item in enumerate(results, start=1):
        W = item["W"]
        if W.size != eeg_data.shape[0]:
            raise ValueError(f"W #{idx} and EEG must have the same number of channels.")
        item["FSResult"] = W @ eeg_data
        print(f"FSResult #{idx} (ERP {item['erp_time']:g} ms): "
              f"shape {item['FSResult'].shape}")
        print("FSResult:", item["FSResult"])
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(time, item["FSResult"], linewidth=0.5)
        ax.set_title(f"Segnale FSResult #{idx} - ERP {item['erp_time']:g} ms")
        ax.set_xlabel("Tempo (secondi)")
        ax.set_ylabel("Ampiezza FSResult")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
        item["fig_FSResult"] = fig
        item["ax_FSResult"] = ax
else:
    print("FS_flag=False: FSResult computation and plots skipped.")

# %% Compare A and W for each ERP center
# Report the coefficient-wise difference and Euclidean distance for every computed A/W pair.

for idx, item in enumerate(results, start=1):
    diff_A_W = item["A"] - item["W"]
    distanza = np.linalg.norm(diff_A_W)
    item["diff_A_W"] = diff_A_W
    item["distanza"] = distanza
    print(f"ERP #{idx} ({item['erp_time']:g} ms):")
    print("Differenza elemento per elemento:", diff_A_W)
    print("Distanza euclidea:", distanza)
