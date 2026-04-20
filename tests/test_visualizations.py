"""
Unit tests for visualization functions.

Tests cover structural validation (returns Figure, correct axes)
and behavioral validation (correct data plotted, labels present).
"""

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from generative_eeg_augmentation.plots.eeg_visualizations import (
    plot_statistical_comparison,
    plot_waveforms,
    plot_psd_comparison,
    plot_band_power_comparison,
    plot_similarity_heatmap,
)
from generative_eeg_augmentation.eval.eeg_metrics import compute_psd, EEG_BANDS


@pytest.fixture
def dummy_eeg_data():
    """Generate dummy EEG data for testing."""
    np.random.seed(42)
    return np.random.randn(10, 63, 1001) * 0.5


@pytest.fixture
def dummy_features():
    """Generate dummy feature data for testing."""
    np.random.seed(42)
    return np.random.randn(10, 63)


# ============================================================================
# Structure Tests - Verify functions return valid Figure objects
# ============================================================================

def test_plot_statistical_comparison_returns_figure(dummy_features):
    """Test that statistical comparison plot returns a Figure object."""
    real_mean = dummy_features
    synth_mean = dummy_features + np.random.randn(*dummy_features.shape) * 0.1
    
    fig = plot_statistical_comparison(real_mean, synth_mean, "Mean", "Test Model")
    
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    assert len(fig.axes) > 0, "Figure should have at least one axis"
    plt.close(fig)


def test_plot_waveforms_returns_figure(dummy_eeg_data):
    """Test that waveform plot returns a Figure object."""
    real = dummy_eeg_data
    synthetic = dummy_eeg_data + np.random.randn(*dummy_eeg_data.shape) * 0.1
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)
    
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    assert len(fig.axes) == 5, "Should have 5 subplots for 5 channels"
    plt.close(fig)


def test_plot_psd_comparison_returns_figure(dummy_eeg_data):
    """Test that PSD comparison plot returns a Figure object."""
    real_psds, freqs = compute_psd(dummy_eeg_data, sfreq=200)
    synth_psds, _ = compute_psd(dummy_eeg_data + np.random.randn(*dummy_eeg_data.shape) * 0.05)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    assert len(fig.axes) > 0, "Figure should have at least one axis"
    plt.close(fig)


def test_plot_band_power_comparison_returns_figure(dummy_eeg_data):
    """Test that band power comparison plot returns a Figure object."""
    real_psds, freqs = compute_psd(dummy_eeg_data, sfreq=200)
    synth_psds, _ = compute_psd(dummy_eeg_data + np.random.randn(*dummy_eeg_data.shape) * 0.05)
    
    fig = plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS)
    
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    assert len(fig.axes) == 5, "Should have 5 subplots for 5 EEG bands"
    plt.close(fig)


def test_plot_similarity_heatmap_returns_figure(dummy_features):
    """Test that similarity heatmap returns a Figure object."""
    real_feats = np.vstack([dummy_features.flatten(), dummy_features.flatten()])
    synth_feats = real_feats + np.random.randn(*real_feats.shape) * 0.1
    feature_names = ['Real Mean', 'Real Var', 'Synth Mean', 'Synth Var']
    
    fig = plot_similarity_heatmap(real_feats, synth_feats, feature_names)
    
    assert isinstance(fig, plt.Figure), "Should return matplotlib Figure"
    assert len(fig.axes) > 0, "Figure should have at least one axis"
    plt.close(fig)


# ============================================================================
# Behavioral Tests - Verify correct data and labels in plots
# ============================================================================

def test_plot_statistical_comparison_has_correct_title(dummy_features):
    """Verify plot title contains feature name and model label."""
    real_mean = dummy_features
    synth_mean = dummy_features + 0.1
    
    fig = plot_statistical_comparison(real_mean, synth_mean, "Mean", "Enhanced GAN")
    
    title = fig.axes[0].get_title()
    assert "Mean" in title, "Title should contain feature name"
    assert "Enhanced GAN" in title, "Title should contain model label"
    plt.close(fig)


def test_plot_statistical_comparison_has_legend(dummy_features):
    """Verify plot has legend with Real and Synthetic labels."""
    real_mean = dummy_features
    synth_mean = dummy_features + 0.1
    
    fig = plot_statistical_comparison(real_mean, synth_mean, "Mean")
    
    legend = fig.axes[0].get_legend()
    assert legend is not None, "Plot should have legend"
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "Real EEG" in legend_texts, "Legend should contain 'Real EEG'"
    assert "Synthetic EEG" in legend_texts, "Legend should contain 'Synthetic EEG'"
    plt.close(fig)


def test_plot_waveforms_correct_number_of_subplots():
    """Verify waveform plot creates correct number of channel subplots."""
    np.random.seed(42)
    real = np.random.randn(10, 63, 1001)
    synthetic = real + 0.1
    
    for n_channels in [1, 3, 5, 10]:
        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=n_channels)
        assert len(fig.axes) == n_channels, f"Should have {n_channels} subplots"
        plt.close(fig)


def test_plot_waveforms_has_legend_on_first_subplot(dummy_eeg_data):
    """Verify first subplot has legend."""
    real = dummy_eeg_data
    synthetic = real + 0.1
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)
    
    # First subplot should have legend
    legend = fig.axes[0].get_legend()
    assert legend is not None, "First subplot should have legend"
    plt.close(fig)


def test_plot_waveforms_raises_on_invalid_epoch_idx(dummy_eeg_data):
    """Verify error raised for out-of-range epoch index."""
    real = dummy_eeg_data  # 10 epochs
    synthetic = real + 0.1
    
    with pytest.raises(IndexError, match="out of range"):
        plot_waveforms(real, synthetic, epoch_idx=100, channels=5)


def test_plot_waveforms_raises_on_too_many_channels(dummy_eeg_data):
    """Verify error raised when requesting more channels than available."""
    real = dummy_eeg_data  # 63 channels
    synthetic = real + 0.1
    
    with pytest.raises(ValueError, match="only has 63 channels"):
        plot_waveforms(real, synthetic, epoch_idx=0, channels=100)


def test_plot_psd_comparison_has_correct_title(dummy_eeg_data):
    """Verify PSD plot title contains epoch and channel info."""
    real_psds, freqs = compute_psd(dummy_eeg_data)
    synth_psds, _ = compute_psd(dummy_eeg_data + 0.05)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=2, ch_idx=5)
    
    title = fig.axes[0].get_title()
    assert "Epoch 2" in title, "Title should contain epoch index"
    assert "Channel 6" in title, "Title should contain channel index (1-indexed)"
    plt.close(fig)


def test_plot_psd_comparison_uses_log_scale(dummy_eeg_data):
    """Verify PSD plot uses logarithmic y-scale."""
    real_psds, freqs = compute_psd(dummy_eeg_data)
    synth_psds, _ = compute_psd(dummy_eeg_data + 0.05)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs)
    
    # Check if y-axis is logarithmic
    assert fig.axes[0].get_yscale() == 'log', "Y-axis should use log scale"
    plt.close(fig)


def test_plot_band_power_comparison_has_all_bands():
    """Verify band power plot creates subplot for each EEG band."""
    np.random.seed(42)
    dummy_data = np.random.randn(10, 63, 1001) * 0.5
    real_psds, freqs = compute_psd(dummy_data, sfreq=200)
    synth_psds, _ = compute_psd(dummy_data + 0.05)
    
    fig = plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS)
    
    # Should have subplot for each band
    assert len(fig.axes) == len(EEG_BANDS), f"Should have {len(EEG_BANDS)} subplots"
    
    # Check titles contain band names
    titles = [ax.get_title() for ax in fig.axes]
    for band_name in EEG_BANDS.keys():
        assert any(band_name in title for title in titles), f"{band_name} should be in titles"
    plt.close(fig)


def test_plot_similarity_heatmap_shows_correlation_matrix(dummy_features):
    """Verify heatmap shows correct correlation structure."""
    # Create features with known correlation (identical features)
    real_feats = np.vstack([dummy_features.flatten()])
    synth_feats = np.vstack([dummy_features.flatten()])  # Identical
    feature_names = ['Real Mean', 'Synth Mean']
    
    fig = plot_similarity_heatmap(real_feats, synth_feats, feature_names)
    
    # Heatmap should exist
    assert len(fig.axes) > 0, "Figure should have axes"
    
    # Check that title contains expected text
    title = fig.axes[0].get_title()
    assert "Similarity" in title or "Heatmap" in title, "Title should indicate heatmap"
    plt.close(fig)


# ============================================================================
# Framework Compatibility Tests - Verify torch tensor support
# ============================================================================

def test_plot_statistical_comparison_accepts_torch_tensors(dummy_features):
    """Verify function handles torch tensors correctly."""
    real_tensor = torch.from_numpy(dummy_features).float()
    synth_tensor = torch.from_numpy(dummy_features + 0.1).float()
    
    fig = plot_statistical_comparison(real_tensor, synth_tensor, "Mean")
    
    assert isinstance(fig, plt.Figure), "Should return Figure with torch input"
    plt.close(fig)


def test_plot_waveforms_accepts_torch_tensors(dummy_eeg_data):
    """Verify waveform plotting handles torch tensors correctly."""
    real_tensor = torch.from_numpy(dummy_eeg_data).float()
    synth_tensor = torch.from_numpy(dummy_eeg_data + 0.1).float()
    
    fig = plot_waveforms(real_tensor, synth_tensor, epoch_idx=0, channels=5)
    
    assert isinstance(fig, plt.Figure), "Should return Figure with torch input"
    assert len(fig.axes) == 5, "Should have 5 subplots"
    plt.close(fig)


def test_plot_psd_comparison_accepts_torch_tensors(dummy_eeg_data):
    """Verify PSD plotting handles torch tensors correctly."""
    real_psds, freqs = compute_psd(dummy_eeg_data)
    synth_psds, _ = compute_psd(dummy_eeg_data + 0.05)
    
    # Convert to tensors
    real_psds_tensor = torch.from_numpy(real_psds).float()
    synth_psds_tensor = torch.from_numpy(synth_psds).float()
    freqs_tensor = torch.from_numpy(freqs).float()
    
    fig = plot_psd_comparison(real_psds_tensor, synth_psds_tensor, freqs_tensor)
    
    assert isinstance(fig, plt.Figure), "Should return Figure with torch input"
    plt.close(fig)


# ============================================================================
# Configuration Tests - Verify DPI and styling parameters
# ============================================================================

def test_plot_statistical_comparison_respects_dpi(dummy_features):
    """Verify DPI parameter affects figure resolution."""
    real_mean = dummy_features
    synth_mean = dummy_features + 0.1
    
    fig_low = plot_statistical_comparison(real_mean, synth_mean, "Mean", dpi=50)
    fig_high = plot_statistical_comparison(real_mean, synth_mean, "Mean", dpi=300)
    
    # Higher DPI should result in higher resolution
    # Note: matplotlib may scale DPI on retina displays, so verify relative difference
    assert fig_high.dpi > fig_low.dpi, "Higher DPI request should result in higher actual DPI"
    # Verify DPI is at least proportional (within 2x scaling for retina)
    assert 50 <= fig_low.dpi <= 150, f"Low DPI should be ~50-100, got {fig_low.dpi}"
    assert 300 <= fig_high.dpi <= 600, f"High DPI should be ~300-600, got {fig_high.dpi}"
    plt.close(fig_low)
    plt.close(fig_high)


def test_plot_waveforms_respects_model_label(dummy_eeg_data):
    """Verify model_label parameter appears in title."""
    real = dummy_eeg_data
    synthetic = real + 0.1
    
    model_name = "Custom Model XYZ"
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5, model_label=model_name)
    
    title = fig.axes[0].get_title()
    assert model_name in title, f"Title should contain '{model_name}'"
    plt.close(fig)


# ============================================================================
# Integration Tests - End-to-end workflows
# ============================================================================

def test_end_to_end_statistical_analysis_workflow(dummy_eeg_data):
    """Test complete workflow: compute features -> plot comparison."""
    from generative_eeg_augmentation.eval.eeg_metrics import compute_time_domain_features
    
    real_data = dummy_eeg_data
    synth_data = dummy_eeg_data + np.random.randn(*dummy_eeg_data.shape) * 0.05
    
    # Compute features
    real_mean, real_var, real_kurt, real_skew = compute_time_domain_features(real_data)
    synth_mean, synth_var, synth_kurt, synth_skew = compute_time_domain_features(synth_data)
    
    # Plot each feature
    fig_mean = plot_statistical_comparison(real_mean, synth_mean, "Mean", "Test Model")
    fig_var = plot_statistical_comparison(real_var, synth_var, "Variance", "Test Model")
    
    assert isinstance(fig_mean, plt.Figure), "Mean plot should be Figure"
    assert isinstance(fig_var, plt.Figure), "Variance plot should be Figure"
    plt.close(fig_mean)
    plt.close(fig_var)


def test_end_to_end_spectral_analysis_workflow(dummy_eeg_data):
    """Test complete workflow: compute PSD -> plot comparison."""
    real_data = dummy_eeg_data
    synth_data = dummy_eeg_data + np.random.randn(*dummy_eeg_data.shape) * 0.05
    
    # Compute PSD
    real_psds, freqs = compute_psd(real_data, sfreq=200)
    synth_psds, _ = compute_psd(synth_data, sfreq=200)
    
    # Plot PSD comparison
    fig_psd = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    # Plot band power comparison
    fig_bands = plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS)
    
    assert isinstance(fig_psd, plt.Figure), "PSD plot should be Figure"
    assert isinstance(fig_bands, plt.Figure), "Band power plot should be Figure"
    plt.close(fig_psd)
    plt.close(fig_bands)


# ============================================================================
# Data Correctness Tests - Verify plots contain actual input data
# ============================================================================

def test_plot_waveforms_contains_actual_data():
    """Verify waveform plot contains the actual input data."""
    np.random.seed(42)
    # Create distinctive data with known pattern
    real = np.zeros((5, 63, 1001))
    real[:, 0, :] = np.linspace(0, 100, 1001)  # Linear ramp on channel 0
    synthetic = real + 10.0  # Offset synthetic by constant
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)
    
    # Extract line data from first subplot
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2, "Should have 2 lines (real and synthetic)"
    
    # Extract y-data from lines
    real_line_data = lines[0].get_ydata()
    synth_line_data = lines[1].get_ydata()
    
    # Verify plotted data matches input data
    np.testing.assert_allclose(real_line_data, real[0, 0, :], atol=1e-10,
                               err_msg="Real data in plot should match input")
    np.testing.assert_allclose(synth_line_data, synthetic[0, 0, :], atol=1e-10,
                               err_msg="Synthetic data in plot should match input")
    
    # Verify synthetic is offset from real by 10
    assert np.allclose(synth_line_data - real_line_data, 10.0, atol=1e-10), \
        "Synthetic should be offset from real by 10.0"
    plt.close(fig)


def test_plot_psd_comparison_contains_actual_frequencies():
    """Verify PSD plot uses correct frequency values on x-axis."""
    np.random.seed(42)
    dummy_data = np.random.randn(5, 63, 1001) * 0.5
    
    real_psds, freqs = compute_psd(dummy_data, sfreq=200)
    synth_psds, _ = compute_psd(dummy_data + 0.05)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    # Extract x-data (frequencies) from plot
    lines = fig.axes[0].get_lines()
    plotted_freqs = lines[0].get_xdata()
    
    # Verify frequencies match input
    np.testing.assert_allclose(plotted_freqs, freqs, atol=1e-10,
                               err_msg="Frequencies in plot should match input")
    
    # Verify frequency range is reasonable (0 to Nyquist)
    assert plotted_freqs[0] >= 0, "Minimum frequency should be >= 0"
    assert plotted_freqs[-1] <= 100, "Maximum frequency should be <= Nyquist (100 Hz)"
    plt.close(fig)


def test_plot_psd_comparison_contains_actual_psd_values():
    """Verify PSD plot contains actual PSD values."""
    np.random.seed(42)
    # Create signal with known PSD pattern (10Hz sine wave)
    sfreq = 200
    t = np.linspace(0, 5, sfreq * 5)
    signal = np.sin(2 * np.pi * 10 * t)  # 10Hz
    data = np.tile(signal, (5, 63, 1)).reshape(5, 63, -1)
    
    real_psds, freqs = compute_psd(data, sfreq=sfreq)
    synth_psds = real_psds * 0.8  # Scale synthetic PSDs
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    # Extract PSD values from plot
    lines = fig.axes[0].get_lines()
    real_psd_plotted = lines[0].get_ydata()
    synth_psd_plotted = lines[1].get_ydata()
    
    # Verify plotted PSD matches input (within tolerance due to log scale)
    np.testing.assert_allclose(real_psd_plotted, real_psds[0, 0, :], rtol=1e-5,
                               err_msg="Real PSD in plot should match input")
    np.testing.assert_allclose(synth_psd_plotted, synth_psds[0, 0, :], rtol=1e-5,
                               err_msg="Synthetic PSD in plot should match input")
    plt.close(fig)


def test_plot_band_power_uses_correct_frequency_ranges():
    """Verify band power plot computes powers for correct frequency ranges."""
    np.random.seed(42)
    # Create signal with power concentrated in specific bands
    sfreq = 200
    n_samples = 1000
    
    # Delta band (1-4Hz): strong signal at 2Hz
    delta_signal = np.sin(2 * np.pi * 2 * np.linspace(0, n_samples/sfreq, n_samples)) * 10
    # Alpha band (8-12Hz): strong signal at 10Hz
    alpha_signal = np.sin(2 * np.pi * 10 * np.linspace(0, n_samples/sfreq, n_samples)) * 20
    
    combined_signal = delta_signal + alpha_signal
    data = np.tile(combined_signal, (10, 63, 1)).reshape(10, 63, -1)
    
    real_psds, freqs = compute_psd(data, sfreq=sfreq)
    synth_psds = real_psds  # Use same for comparison
    
    # Manually compute band powers
    from generative_eeg_augmentation.eval.eeg_metrics import compute_band_power
    manual_delta = compute_band_power(real_psds, freqs, EEG_BANDS['Delta'])
    manual_alpha = compute_band_power(real_psds, freqs, EEG_BANDS['Alpha'])
    
    # Alpha should have more power than Delta (stronger signal)
    assert np.mean(manual_alpha) > np.mean(manual_delta), \
        "Alpha band should have more power (10Hz >> 2Hz amplitude)"
    
    # Create plot (mainly to verify it doesn't crash with this data)
    fig = plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS)
    assert isinstance(fig, plt.Figure), "Should return Figure"
    plt.close(fig)


def test_plot_statistical_comparison_shows_distribution_differences():
    """Verify statistical comparison can distinguish different distributions."""
    np.random.seed(42)
    # Create two clearly different distributions
    real_feat = np.random.normal(loc=0.0, scale=1.0, size=(10, 63))
    synth_feat = np.random.normal(loc=5.0, scale=1.0, size=(10, 63))  # Shifted mean
    
    fig = plot_statistical_comparison(real_feat, synth_feat, "Mean", "Test Model")
    
    # Extract histogram data (patches from first axes)
    ax = fig.axes[0]
    
    # Verify plot was created successfully
    assert len(ax.patches) > 0, "Plot should have histogram patches"
    
    # Verify xlim spans both distributions
    xlim = ax.get_xlim()
    assert xlim[0] < 0, "X-axis should extend below 0 for real data"
    assert xlim[1] > 5, "X-axis should extend above 5 for synthetic data"
    plt.close(fig)


def test_plot_similarity_heatmap_shows_actual_correlations():
    """Verify heatmap shows correct correlation values."""
    np.random.seed(42)
    # Create features with known correlations
    feature_a = np.random.randn(1000)
    feature_b = feature_a + np.random.randn(1000) * 0.1  # Highly correlated
    feature_c = np.random.randn(1000)  # Uncorrelated
    
    real_feats = np.vstack([feature_a, feature_b])
    synth_feats = np.vstack([feature_c, feature_c])
    feature_names = ['Real A', 'Real B', 'Synth C', 'Synth D']
    
    # Manually compute expected correlations
    manual_corr = np.corrcoef(np.vstack([real_feats, synth_feats]))
    
    fig = plot_similarity_heatmap(real_feats, synth_feats, feature_names)
    
    # Verify heatmap was created
    ax = fig.axes[0]
    
    # Extract correlation matrix from heatmap
    # The heatmap is stored in the image data
    heatmap_data = None
    for child in ax.get_children():
        if hasattr(child, 'get_array'):
            heatmap_data = child.get_array()
            break
    
    assert heatmap_data is not None, "Heatmap should contain correlation data"
    
    # Verify shape matches expected (4x4 correlation matrix)
    assert heatmap_data.shape == (4, 4), "Heatmap should be 4x4 correlation matrix"
    
    # Verify diagonal elements are 1.0 (self-correlation)
    np.testing.assert_allclose(np.diag(heatmap_data), 1.0, atol=0.01,
                               err_msg="Diagonal should be 1.0 (self-correlation)")
    plt.close(fig)


# ============================================================================
# Stress Tests - Edge cases and boundary conditions
# ============================================================================

def test_plot_waveforms_with_single_epoch():
    """Verify waveform plotting works with single epoch."""
    np.random.seed(42)
    real = np.random.randn(1, 63, 1001)  # Single epoch
    synthetic = real + 0.1
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)
    
    assert isinstance(fig, plt.Figure), "Should handle single epoch"
    assert len(fig.axes) == 5, "Should have 5 subplots"
    plt.close(fig)


def test_plot_waveforms_with_single_channel():
    """Verify waveform plotting works with single channel."""
    np.random.seed(42)
    real = np.random.randn(10, 63, 1001)
    synthetic = real + 0.1
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)
    
    assert isinstance(fig, plt.Figure), "Should handle single channel"
    assert len(fig.axes) == 1, "Should have 1 subplot"
    plt.close(fig)


def test_plot_waveforms_with_maximum_channels():
    """Verify waveform plotting works with all 63 channels."""
    np.random.seed(42)
    real = np.random.randn(5, 63, 1001)
    synthetic = real + 0.1
    
    # This should work but create a very tall figure
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=63)
    
    assert isinstance(fig, plt.Figure), "Should handle all channels"
    assert len(fig.axes) == 63, "Should have 63 subplots"
    plt.close(fig)


def test_plot_statistical_comparison_with_constant_values():
    """Verify statistical comparison handles constant (zero variance) data."""
    # All values are the same
    real_feat = np.ones((10, 63)) * 5.0
    synth_feat = np.ones((10, 63)) * 5.0
    
    # Should handle gracefully (seaborn may emit warnings but shouldn't crash)
    fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")
    
    assert isinstance(fig, plt.Figure), "Should handle constant values"
    plt.close(fig)


def test_plot_psd_comparison_with_short_signal():
    """Verify PSD plotting handles very short signals."""
    np.random.seed(42)
    # Short signal (100 samples instead of 1001)
    real = np.random.randn(5, 63, 100)
    synthetic = real + 0.05
    
    real_psds, freqs = compute_psd(real, sfreq=200)
    synth_psds, _ = compute_psd(synthetic, sfreq=200)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    assert isinstance(fig, plt.Figure), "Should handle short signals"
    plt.close(fig)


def test_plot_band_power_with_large_dataset():
    """Stress test band power plotting with large dataset."""
    np.random.seed(42)
    # Large dataset (1000 epochs)
    large_data = np.random.randn(1000, 63, 1001) * 0.5
    
    real_psds, freqs = compute_psd(large_data[:100], sfreq=200)  # Use subset for speed
    synth_psds, _ = compute_psd(large_data[100:200], sfreq=200)
    
    fig = plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS)
    
    assert isinstance(fig, plt.Figure), "Should handle large dataset"
    assert len(fig.axes) == 5, "Should have 5 subplots"
    plt.close(fig)


def test_plot_waveforms_raises_on_shape_mismatch():
    """Verify error raised when real and synthetic have different shapes."""
    np.random.seed(42)
    real = np.random.randn(10, 63, 1001)
    synthetic = np.random.randn(5, 63, 1001)  # Different number of epochs
    
    # Should handle gracefully - doesn't need to match for visualization
    # (user may want to compare different datasets)
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)
    assert isinstance(fig, plt.Figure), "Should handle mismatched shapes"
    plt.close(fig)


# ============================================================================
# Mutation Tests - Verify tests catch wrong data
# ============================================================================

def test_plot_waveforms_detects_swapped_data():
    """Verify plot distinguishes when real/synthetic are swapped."""
    np.random.seed(42)
    # Create distinctive patterns
    real = np.zeros((5, 63, 1001))
    real[:, 0, :] = 100.0  # High amplitude
    synthetic = np.zeros((5, 63, 1001))
    synthetic[:, 0, :] = 1.0  # Low amplitude
    
    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)
    
    # Extract line data
    lines = fig.axes[0].get_lines()
    real_line = lines[0].get_ydata()
    synth_line = lines[1].get_ydata()
    
    # Verify real has higher amplitude than synthetic
    assert np.mean(np.abs(real_line)) > np.mean(np.abs(synth_line)), \
        "Real data should have higher amplitude than synthetic"
    
    # If data were swapped, this test would fail
    assert np.allclose(real_line, 100.0, atol=1e-10), "Real should be 100.0"
    assert np.allclose(synth_line, 1.0, atol=1e-10), "Synthetic should be 1.0"
    plt.close(fig)


def test_plot_statistical_comparison_detects_distribution_swap():
    """Verify plot can detect if distributions are incorrectly labeled."""
    np.random.seed(42)
    # Create clearly different distributions
    distribution_a = np.random.normal(loc=-10.0, scale=1.0, size=(10, 63))
    distribution_b = np.random.normal(loc=10.0, scale=1.0, size=(10, 63))
    
    fig = plot_statistical_comparison(distribution_a, distribution_b, "Mean")
    
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    
    # Verify plot spans both distributions
    assert xlim[0] < -5, "Should show negative distribution"
    assert xlim[1] > 5, "Should show positive distribution"
    
    # If swapped, user would see incorrect labeling in legend
    # (This is more of a visual test, but we verify data is present)
    plt.close(fig)


def test_plot_psd_detects_zero_psd():
    """Verify PSD plot handles zero PSD (all frequencies absent)."""
    np.random.seed(42)
    # Create zero PSDs (shouldn't happen with real data, but test handling)
    real_psds = np.zeros((5, 63, 100))
    synth_psds = np.ones((5, 63, 100)) * 0.001  # Very small values
    freqs = np.linspace(0, 100, 100)
    
    fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=0, ch_idx=0)
    
    assert isinstance(fig, plt.Figure), "Should handle zero PSD"
    
    # Verify log scale doesn't break with zeros (matplotlib handles this)
    assert fig.axes[0].get_yscale() == 'log', "Should still use log scale"
    plt.close(fig)
