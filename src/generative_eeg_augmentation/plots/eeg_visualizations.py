"""
Visualization functions for EEG data analysis.

This module provides functions to create matplotlib figures for comparing
real and synthetic EEG data across multiple domains (time, frequency, statistics).
All functions return matplotlib Figure objects for flexible display in notebooks
or embedding in applications like Streamlit.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple, List, Union
import torch

# Import band power computation and EEG bands from eval module
from generative_eeg_augmentation.eval.eeg_metrics import compute_band_power, EEG_BANDS


def set_eeg_visualization_style():
    """
    Apply recommended visualization style for EEG plots.

    Sets seaborn theme with whitegrid style, deep color palette, and
    increased font scale for better readability. Call this function
    explicitly if you want consistent styling across all EEG plots.

    This function modifies global matplotlib/seaborn settings and will
    affect all subsequent plots in the session.

    Example:
        >>> from generative_eeg_augmentation.plots import set_eeg_visualization_style
        >>> set_eeg_visualization_style()
        >>> # All subsequent plots will use the EEG style
    """
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert torch.Tensor to numpy array if needed.

    Ensures the tensor is on CPU, detached from the computation graph,
    and contiguous in memory before conversion to numpy.

    Args:
        data: Input data (numpy array or torch tensor).

    Returns:
        numpy array (contiguous).
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().contiguous().numpy()
    return np.ascontiguousarray(data)


def plot_statistical_comparison(
    real_feat: Union[np.ndarray, torch.Tensor],
    synth_feat: Union[np.ndarray, torch.Tensor],
    feat_name: str,
    model_label: str = "Model",
    dpi: int = 100
) -> plt.Figure:
    """
    Plot histogram comparison of a statistical feature between real and synthetic data.

    Creates overlaid histograms with KDE curves to compare the distribution
    of a statistical feature (mean, variance, kurtosis, skewness) between
    real and synthetic EEG data.

    Args:
        real_feat: Feature values from real data (any shape, will be flattened).
        synth_feat: Feature values from synthetic data (any shape, will be flattened).
        feat_name: Name of the feature (e.g., "Mean", "Variance", "Kurtosis").
        model_label: Label for the model variant (default "Model").
        dpi: Figure DPI (default 100 for speed, 300 for publication).

    Returns:
        matplotlib Figure object.

    Example:
        >>> real_mean = np.random.randn(630)
        >>> synth_mean = np.random.randn(630)
        >>> fig = plot_statistical_comparison(real_mean, synth_mean, "Mean", "CWGAN")
        >>> plt.show()  # or display in notebook
    """
    # Convert to numpy if needed
    real_feat = _to_numpy(real_feat)
    synth_feat = _to_numpy(synth_feat)

    fig = plt.figure(figsize=(12, 5), dpi=dpi)
    sns.histplot(real_feat.flatten(), kde=True, color='blue',
                 label='Real EEG', alpha=0.6)
    sns.histplot(synth_feat.flatten(), kde=True, color='orange',
                 label='Synthetic EEG', alpha=0.6)
    plt.title(f"{feat_name} Distribution: Real vs Synthetic EEG ({model_label})")
    plt.xlabel(feat_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_waveforms(
    real: Union[np.ndarray, torch.Tensor],
    synthetic: Union[np.ndarray, torch.Tensor],
    epoch_idx: int,
    channels: int = 5,
    model_label: str = "Model",
    dpi: int = 100
) -> plt.Figure:
    """
    Plot time-domain waveform comparison for multiple channels.

    Creates a multi-panel figure showing real vs synthetic EEG signals
    for a specific epoch across multiple channels.

    Args:
        real: Real EEG data, shape (n_epochs, n_channels, n_timepoints).
        synthetic: Synthetic EEG data, same shape as real.
        epoch_idx: Index of epoch to visualize.
        channels: Number of channels to display (default 5).
        model_label: Label for the model variant (default "Model").
        dpi: Figure DPI (default 100 for speed, 300 for publication).

    Returns:
        matplotlib Figure object.

    Example:
        >>> real_data = np.random.randn(10, 63, 1001)
        >>> synth_data = np.random.randn(10, 63, 1001)
        >>> fig = plot_waveforms(real_data, synth_data, epoch_idx=0, channels=5)
        >>> plt.show()
    """
    # Convert to numpy if needed
    real = _to_numpy(real)
    synthetic = _to_numpy(synthetic)

    fig = plt.figure(figsize=(14, 8), dpi=dpi)
    for ch in range(channels):
        plt.subplot(channels, 1, ch + 1)
        plt.plot(real[epoch_idx, ch], label="Real EEG", linewidth=1)
        plt.plot(synthetic[epoch_idx, ch], linestyle='--',
                 label="Synthetic EEG", linewidth=1, alpha=0.8)
        plt.ylabel(f"Ch {ch + 1}")
        if ch == 0:
            plt.title(f"Real vs Synthetic EEG Signals (Epoch {epoch_idx}) ({model_label})")
            plt.legend()
        if ch == channels - 1:
            plt.xlabel("Time (samples)")
    plt.tight_layout()
    return fig


def plot_psd_comparison(
    real_psd: np.ndarray,
    synth_psd: np.ndarray,
    freqs: np.ndarray,
    epoch_idx: int = 0,
    ch_idx: int = 0,
    dpi: int = 100
) -> plt.Figure:
    """
    Plot PSD comparison for a single epoch and channel.

    Creates a semi-logarithmic plot comparing power spectral density
    between real and synthetic EEG for a specific epoch and channel.

    Args:
        real_psd: Real PSD, shape (n_epochs, n_channels, n_frequencies).
        synth_psd: Synthetic PSD, same shape as real_psd.
        freqs: Frequency values corresponding to PSD.
        epoch_idx: Epoch index to visualize (default 0).
        ch_idx: Channel index to visualize (default 0).
        dpi: Figure DPI (default 100 for speed, 300 for publication).

    Returns:
        matplotlib Figure object.

    Example:
        >>> from generative_eeg_augmentation.eval.eeg_metrics import compute_psd
        >>> data = np.random.randn(10, 63, 1001)
        >>> real_psd, freqs = compute_psd(data)
        >>> synth_psd, _ = compute_psd(data + np.random.randn(*data.shape) * 0.1)
        >>> fig = plot_psd_comparison(real_psd, synth_psd, freqs, epoch_idx=0, ch_idx=0)
        >>> plt.show()
    """
    fig = plt.figure(figsize=(12, 5), dpi=dpi)
    plt.semilogy(freqs, real_psd[epoch_idx, ch_idx], label='Real EEG', linewidth=2)
    plt.semilogy(freqs, synth_psd[epoch_idx, ch_idx], label='Synthetic EEG',
                 linewidth=2, alpha=0.8)
    plt.title(f"PSD Comparison (Epoch {epoch_idx}, Channel {ch_idx + 1})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V²/Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_band_power_comparison(
    real_psds: np.ndarray,
    synthetic_psds: np.ndarray,
    freqs: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    dpi: int = 100
) -> plt.Figure:
    """
    Plot band power distribution comparison across all standard EEG bands.

    Creates a multi-panel figure showing histogram comparisons for each
    EEG frequency band (Delta, Theta, Alpha, Beta, Gamma).

    Args:
        real_psds: Real PSD data, shape (n_epochs, n_channels, n_frequencies).
        synthetic_psds: Synthetic PSD data, same shape as real_psds.
        freqs: Frequency values corresponding to PSD.
        bands: Dictionary of band names to (fmin, fmax) tuples.
               If None, uses standard EEG_BANDS.
        dpi: Figure DPI (default 100 for speed, 300 for publication).

    Returns:
        matplotlib Figure object.

    Example:
        >>> from generative_eeg_augmentation.eval.eeg_metrics import compute_psd, EEG_BANDS
        >>> data = np.random.randn(10, 63, 1001)
        >>> real_psd, freqs = compute_psd(data)
        >>> synth_psd, _ = compute_psd(data + np.random.randn(*data.shape) * 0.1)
        >>> fig = plot_band_power_comparison(real_psd, synth_psd, freqs, EEG_BANDS)
        >>> plt.show()
    """
    if bands is None:
        bands = EEG_BANDS

    # Calculate grid dimensions based on number of bands
    n_bands = len(bands)
    n_cols = 2
    n_rows = int(np.ceil(n_bands / n_cols))

    fig = plt.figure(figsize=(12, 5 * n_rows), dpi=dpi)

    for idx, (band_name, freq_range) in enumerate(bands.items(), 1):
        real_band = compute_band_power(real_psds, freqs, freq_range)
        synth_band = compute_band_power(synthetic_psds, freqs, freq_range)

        plt.subplot(n_rows, n_cols, idx)
        sns.histplot(real_band.flatten(), kde=True, label="Real EEG",
                     color='blue', alpha=0.6)
        sns.histplot(synth_band.flatten(), kde=True, label="Synthetic EEG",
                     color='orange', alpha=0.6)
        plt.title(f"{band_name} Band ({freq_range[0]}-{freq_range[1]} Hz)")
        plt.xlabel("Band Power")
        plt.ylabel("Density")
        if idx == 1:
            plt.legend()

    plt.tight_layout()
    return fig


def plot_similarity_heatmap(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    feature_names: List[str],
    model_label: str = "Model",
    dpi: int = 100
) -> plt.Figure:
    """
    Plot correlation heatmap between real and synthetic statistical features.

    Creates a heatmap showing the correlation matrix of combined real and
    synthetic features to visualize statistical similarity.

    Args:
        real_features: Real feature matrix, shape (n_features, n_samples).
        synthetic_features: Synthetic feature matrix, same shape as real_features.
        feature_names: Names for each feature (e.g., ["Mean", "Variance", "Kurtosis", "Skewness"]).
        model_label: Label for the model variant (default "Model").
        dpi: Figure DPI (default 100 for speed, 300 for publication).

    Returns:
        matplotlib Figure object.

    Example:
        >>> real_feats = np.random.randn(4, 630)  # 4 features, 630 samples
        >>> synth_feats = np.random.randn(4, 630)
        >>> feature_names = ["Mean", "Variance", "Kurtosis", "Skewness"]
        >>> fig = plot_similarity_heatmap(real_feats, synth_feats, feature_names, "CWGAN")
        >>> plt.show()
    """
    combined_feats = np.vstack([real_features, synthetic_features])
    similarity_matrix = np.corrcoef(combined_feats)

    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})

    plt.title(f"Heatmap of Statistical Similarity (Real vs Synthetic EEG) ({model_label})",
              fontsize=16, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    return fig
