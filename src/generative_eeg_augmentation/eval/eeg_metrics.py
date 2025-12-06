"""
Evaluation metrics for EEG data quality assessment.

This module provides functions to compute time-domain and frequency-domain
metrics for comparing real and synthetic EEG data.
"""

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from typing import Tuple, Dict, Union


# Standard EEG frequency bands (Hz)
EEG_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 40)
}


def compute_time_domain_features(
    data: Union[torch.Tensor, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistical features in time domain for EEG data.
    
    Calculates mean, variance, kurtosis, and skewness across the time
    dimension for each epoch and channel.
    
    Args:
        data: EEG data of shape (n_epochs, n_channels, n_timepoints).
            Can be torch.Tensor or np.ndarray.
    
    Returns:
        Tuple of (mean, variance, kurtosis, skewness), each with shape
        (n_epochs, n_channels).
    
    Example:
        >>> data = np.random.randn(10, 63, 1001)
        >>> mean, var, kurt, skew_val = compute_time_domain_features(data)
        >>> mean.shape
        (10, 63)
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    mean_val = np.mean(data_np, axis=2)
    var_val = np.var(data_np, axis=2)
    kurtosis_val = kurtosis(data_np, axis=2)
    skewness_val = skew(data_np, axis=2)
    
    return mean_val, var_val, kurtosis_val, skewness_val


def compute_psd(
    data: Union[torch.Tensor, np.ndarray],
    sfreq: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Applies Welch's method with 256-sample segments to estimate the
    power spectral density for each epoch and channel.
    
    Args:
        data: EEG data of shape (n_epochs, n_channels, n_timepoints).
        sfreq: Sampling frequency in Hz (default 200).
    
    Returns:
        Tuple of (psds, freqs):
            - psds: np.ndarray of shape (n_epochs, n_channels, n_frequencies)
            - freqs: np.ndarray of frequency values
    
    Example:
        >>> data = np.random.randn(10, 63, 1001)
        >>> psds, freqs = compute_psd(data, sfreq=200)
        >>> psds.shape
        (10, 63, 129)
        >>> freqs.shape
        (129,)
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().detach().numpy()
    else:
        data_np = data
    
    epochs, channels, samples = data_np.shape
    psds = []
    
    for epoch in range(epochs):
        epoch_psds = []
        for ch in range(channels):
            freqs, psd = welch(data_np[epoch, ch, :], fs=sfreq, nperseg=256)
            epoch_psds.append(psd)
        psds.append(epoch_psds)
    
    psds = np.array(psds)  # shape: (epochs, channels, frequencies)
    return psds, freqs


def compute_band_power(
    psds: np.ndarray,
    freqs: np.ndarray,
    band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute average power in a specific frequency band.
    
    Args:
        psds: Power spectral densities, shape (n_epochs, n_channels, n_frequencies).
        freqs: Frequency values corresponding to PSD.
        band: Tuple of (fmin, fmax) defining the frequency band in Hz.
    
    Returns:
        Band power averaged over frequency dimension, shape (n_epochs, n_channels).
    
    Example:
        >>> psds = np.random.rand(10, 63, 129)
        >>> freqs = np.linspace(0, 100, 129)
        >>> alpha_power = compute_band_power(psds, freqs, (8, 12))
        >>> alpha_power.shape
        (10, 63)
    """
    fmin, fmax = band
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.mean(psds[:, :, idx_band], axis=-1)


def compute_all_band_powers(
    psds: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute power for all standard EEG bands.
    
    Args:
        psds: Power spectral densities.
        freqs: Frequency values.
        bands: Dictionary of band names to (fmin, fmax). If None, uses EEG_BANDS.
    
    Returns:
        Dictionary mapping band names to band power arrays of shape (n_epochs, n_channels).
    
    Example:
        >>> psds = np.random.rand(10, 63, 129)
        >>> freqs = np.linspace(0, 100, 129)
        >>> band_powers = compute_all_band_powers(psds, freqs)
        >>> list(band_powers.keys())
        ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        >>> band_powers['Alpha'].shape
        (10, 63)
    """
    if bands is None:
        bands = EEG_BANDS
    
    result = {}
    for band_name, freq_range in bands.items():
        result[band_name] = compute_band_power(psds, freqs, freq_range)
    
    return result
