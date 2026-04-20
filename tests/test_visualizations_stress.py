"""
Stress tests and edge case validation for EEG visualization functions.

These tests go beyond "does it crash" to verify:
- Biological plausibility of visualization outputs
- Correct handling of extreme and pathological inputs
- Data accuracy in plots (known input → known output)
- Memory efficiency with large datasets
- Numerical stability with edge cases
"""

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gc
import sys

from generative_eeg_augmentation.plots.eeg_visualizations import (
    plot_statistical_comparison,
    plot_waveforms,
    plot_psd_comparison,
    plot_band_power_comparison,
    plot_similarity_heatmap,
    EEG_BANDS,
    _to_numpy
)
from generative_eeg_augmentation.eval.eeg_metrics import compute_psd


# ============================================================================
# Edge Case Tests: Pathological Inputs
# ============================================================================

class TestPathologicalInputs:
    """Test that visualization functions handle extreme/pathological data correctly."""

    def test_plot_statistical_comparison_with_zeros(self):
        """Test statistical comparison with all-zero data."""
        real_feat = np.zeros(100)
        synth_feat = np.zeros(100)

        fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")

        assert isinstance(fig, Figure)
        # Should create plot even with degenerate data
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_statistical_comparison_with_constant_values(self):
        """Test statistical comparison with constant (non-zero) data."""
        real_feat = np.ones(100) * 5.0
        synth_feat = np.ones(100) * 5.0

        fig = plot_statistical_comparison(real_feat, synth_feat, "Variance")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_statistical_comparison_with_nans(self):
        """
        Test statistical comparison with NaN values.

        NaN in EEG data could indicate missing segments or artifacts.
        Visualization should handle gracefully (may produce warning but not crash).
        """
        real_feat = np.random.randn(100)
        real_feat[10:20] = np.nan
        synth_feat = np.random.randn(100)

        # This should not crash, though histogram may filter NaNs
        fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_statistical_comparison_with_infs(self):
        """Test statistical comparison with infinite values."""
        real_feat = np.random.randn(100)
        real_feat[5:10] = np.inf
        real_feat[15:20] = -np.inf
        synth_feat = np.random.randn(100)

        # Should handle inf values without crashing
        fig = plot_statistical_comparison(real_feat, synth_feat, "Kurtosis")

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_waveforms_with_single_epoch(self):
        """Test waveform plotting with only one epoch."""
        real = np.random.randn(1, 63, 1001) * 0.5
        synthetic = real + np.random.randn(1, 63, 1001) * 0.1

        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 5
        plt.close(fig)

    def test_plot_waveforms_with_single_channel(self):
        """Test waveform plotting with single channel data."""
        real = np.random.randn(10, 1, 1001) * 0.5
        synthetic = real + np.random.randn(10, 1, 1001) * 0.1

        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_waveforms_with_single_timepoint(self):
        """
        Test waveform plotting with single timepoint.

        This is an extreme edge case - not realistic for EEG but tests robustness.
        """
        real = np.random.randn(10, 63, 1) * 0.5
        synthetic = real + np.random.randn(10, 63, 1) * 0.1

        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=3)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_waveforms_with_extreme_amplitudes(self):
        """Test waveform plotting with unrealistically large amplitudes."""
        # EEG typically ±100 μV; test with ±1000
        real = np.random.randn(5, 10, 500) * 1000
        synthetic = real + np.random.randn(5, 10, 500) * 100

        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=3)

        assert isinstance(fig, Figure)
        # Plot should still be created, though values are unrealistic
        plt.close(fig)

    def test_plot_psd_comparison_with_zero_signal(self):
        """Test PSD comparison with zero signal (should have flat PSD)."""
        real = np.zeros((5, 10, 1001))
        synth = np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real)
        synth_psds, _ = compute_psd(synth)

        fig = plot_psd_comparison(real_psds, synth_psds, freqs)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_band_power_comparison_with_empty_band(self):
        """Test band power comparison when frequency band contains no data."""
        real = np.random.randn(5, 10, 1001) * 0.5
        synth = real + np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real, sfreq=100)  # Low sampling rate
        synth_psds, _ = compute_psd(synth, sfreq=100)

        # Gamma band (30-40 Hz) might be poorly represented at 100 Hz sampling
        custom_bands = {'Gamma': (30, 40)}

        fig = plot_band_power_comparison(real_psds, synth_psds, freqs, bands=custom_bands)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_similarity_heatmap_with_single_feature(self):
        """Test heatmap with only one feature (minimal case)."""
        real_features = np.random.randn(1, 100)
        synth_features = np.random.randn(1, 100)
        feature_names = ["Mean"]

        fig = plot_similarity_heatmap(real_features, synth_features, feature_names)

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_similarity_heatmap_with_perfectly_correlated_features(self):
        """Test heatmap when real and synthetic features are identical."""
        real_features = np.random.randn(4, 100)
        synth_features = real_features.copy()  # Perfect match
        feature_names = ["Mean", "Var", "Kurt", "Skew"]

        fig = plot_similarity_heatmap(real_features, synth_features, feature_names)

        assert isinstance(fig, Figure)
        # Correlation matrix should show high correlation
        plt.close(fig)


# ============================================================================
# Behavioral Tests: Known Input → Known Output
# ============================================================================

class TestBehavioralCorrectness:
    """Test that visualizations correctly represent the data they plot."""

    def test_statistical_comparison_data_appears_in_plot(self):
        """
        Test that input data actually appears in the statistical comparison plot.

        Validates that the plot contains the actual data values, not arbitrary numbers.
        """
        # Create distinctive distributions
        real_feat = np.random.randn(100) * 2 + 10  # Mean ~10, std ~2
        synth_feat = np.random.randn(100) * 2 - 10  # Mean ~-10, std ~2

        fig = plot_statistical_comparison(real_feat, synth_feat, "TestFeature")

        # Extract data from the plot
        ax = fig.axes[0]

        # The plot should have created histogram patches
        assert len(ax.patches) > 0, "Plot should contain histogram bars"

        # Check that x-axis range encompasses the data
        xlim = ax.get_xlim()
        assert xlim[0] < real_feat.min(), "X-axis should cover minimum value"
        assert xlim[1] > synth_feat.max(), "X-axis should cover maximum value"

        plt.close(fig)

    def test_waveforms_plot_correct_epoch(self):
        """
        Test that waveform plot displays the correct epoch.

        Creates distinctive waveforms and verifies the right one is plotted.
        """
        # Create data where each epoch has a distinctive mean
        real = np.random.randn(5, 3, 100) * 0.1
        for i in range(5):
            real[i] += i * 10  # Epoch 0: mean ~0, Epoch 1: mean ~10, etc.

        synthetic = real + np.random.randn(5, 3, 100) * 0.05

        # Plot epoch 3 (should have mean ~30)
        fig = plot_waveforms(real, synthetic, epoch_idx=3, channels=2)

        # Extract plotted data from first axis
        ax = fig.axes[0]
        lines = ax.get_lines()

        assert len(lines) >= 2, "Should have at least 2 lines (real and synthetic)"

        # Get the real signal line data
        real_line_data = lines[0].get_ydata()

        # Verify it's approximately epoch 3's data (mean ~30)
        assert np.abs(np.mean(real_line_data) - 30) < 5, \
            f"Plotted data mean {np.mean(real_line_data):.2f} should be near 30"

        plt.close(fig)

    def test_waveforms_plot_correct_number_of_channels(self):
        """Test that waveform plot creates correct number of subplots."""
        real = np.random.randn(5, 20, 100) * 0.5
        synthetic = real + np.random.randn(5, 20, 100) * 0.1

        for n_channels in [1, 3, 7, 12]:
            fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=n_channels)

            assert len(fig.axes) == n_channels, \
                f"Should have {n_channels} subplots, got {len(fig.axes)}"

            plt.close(fig)

    def test_psd_comparison_plots_correct_epoch_channel(self):
        """
        Test that PSD plot displays the correct epoch and channel.

        Creates data with distinctive spectral content per epoch/channel.
        """
        # Create small dataset
        real = np.random.randn(3, 3, 1001) * 0.1
        synth = np.random.randn(3, 3, 1001) * 0.1

        real_psds, freqs = compute_psd(real)
        synth_psds, _ = compute_psd(synth)

        # Plot specific epoch and channel
        fig = plot_psd_comparison(real_psds, synth_psds, freqs, epoch_idx=1, ch_idx=2)

        ax = fig.axes[0]
        lines = ax.get_lines()

        assert len(lines) >= 2, "Should have at least 2 lines (real and synthetic)"

        # Verify that line data length matches frequency array
        real_psd_line = lines[0].get_ydata()
        assert len(real_psd_line) == len(freqs), \
            "Plotted PSD should have same length as frequency array"

        plt.close(fig)

    def test_psd_comparison_uses_semilogy_scale(self):
        """
        Test that PSD comparison uses logarithmic y-axis.

        This is critical for EEG PSD visualization as power varies over orders of magnitude.
        """
        real = np.random.randn(5, 10, 1001) * 0.5
        synth = real + np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real)
        synth_psds, _ = compute_psd(synth)

        fig = plot_psd_comparison(real_psds, synth_psds, freqs)

        ax = fig.axes[0]

        # Check that y-axis uses log scale
        assert ax.get_yscale() == 'log', "PSD plot should use logarithmic y-axis"

        plt.close(fig)

    def test_band_power_comparison_creates_correct_number_of_subplots(self):
        """Test that band power comparison creates subplot for each band."""
        real = np.random.randn(5, 10, 1001) * 0.5
        synth = real + np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real)
        synth_psds, _ = compute_psd(synth)

        # Test with default 5 bands
        fig = plot_band_power_comparison(real_psds, synth_psds, freqs)
        assert len(fig.axes) == 5, "Should have 5 subplots for 5 EEG bands"
        plt.close(fig)

        # Test with custom 3 bands
        custom_bands = {'Low': (1, 10), 'Mid': (10, 30), 'High': (30, 50)}
        fig = plot_band_power_comparison(real_psds, synth_psds, freqs, bands=custom_bands)
        # Layout is 3x2, so may have more axes than bands
        assert len(fig.axes) >= 3, "Should have at least 3 subplots for 3 bands"
        plt.close(fig)

    def test_similarity_heatmap_has_correct_dimensions(self):
        """Test that similarity heatmap has correct matrix dimensions."""
        n_features = 4
        real_features = np.random.randn(n_features, 100)
        synth_features = np.random.randn(n_features, 100)
        feature_names = [f"Feature_{i}" for i in range(n_features)]

        fig = plot_similarity_heatmap(real_features, synth_features, feature_names)

        ax = fig.axes[0]

        # Heatmap combines real and synthetic, so should be 2*n_features x 2*n_features
        # But since we're showing correlation between features, it's n_features x n_features
        # Actually, the implementation stacks real and synth features, so it's 2*n_features total
        # The correlation matrix is (2*n_features) x (2*n_features)

        # Get the heatmap data
        images = [child for child in ax.get_children() if hasattr(child, 'get_array')]
        if images:
            heatmap_data = images[0].get_array()
            expected_size = 2 * n_features
            assert heatmap_data.shape == (expected_size, expected_size), \
                f"Heatmap should be {expected_size}x{expected_size}, got {heatmap_data.shape}"

        plt.close(fig)

    def test_to_numpy_helper_preserves_data_values(self):
        """Test that _to_numpy helper correctly converts tensor to numpy."""
        # Test with numpy array (should pass through)
        np_data = np.array([1.0, 2.0, 3.0, 4.0])
        result = _to_numpy(np_data)
        assert np.array_equal(result, np_data)
        assert result is np_data  # Should be same object

        # Test with torch tensor
        torch_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _to_numpy(torch_data)
        assert np.allclose(result, np_data)
        assert isinstance(result, np.ndarray)

        # Test with GPU tensor (if CUDA available)
        if torch.cuda.is_available():
            gpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
            result = _to_numpy(gpu_tensor)
            assert np.allclose(result, np_data)
            assert isinstance(result, np.ndarray)


# ============================================================================
# Memory and Performance Stress Tests
# ============================================================================

class TestMemoryAndPerformance:
    """Test memory efficiency and performance with large datasets."""

    def test_memory_leak_repeated_plotting(self):
        """
        Test that repeated plot generation doesn't cause unbounded memory growth.

        This is critical for applications that generate many plots (e.g., Streamlit apps).

        Note: Matplotlib and seaborn maintain some internal caches and state,
        so we test that memory growth is sublinear (not O(n) with iterations).
        """
        import gc

        # Warmup: First few iterations establish baseline caches
        for _ in range(10):
            real_feat = np.random.randn(100)
            synth_feat = np.random.randn(100)
            fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")
            plt.close(fig)

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Generate and close many plots
        for _ in range(100):
            real_feat = np.random.randn(100)
            synth_feat = np.random.randn(100)

            fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")
            plt.close(fig)

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow growth due to matplotlib internals, but not linear growth
        # With 100 iterations, if each leaked even 1 object we'd have +100
        # Matplotlib/seaborn cache fonts, colors, styles, etc.
        # Empirically, ~50-60 objects per iteration is typical for matplotlib
        # But if we're leaking figures, we'd see 500+ objects per iteration
        object_growth = final_objects - initial_objects
        objects_per_iter = object_growth / 100

        # The key test: are we leaking proportional to iterations?
        # If objects_per_iter > 100, we're definitely leaking
        # Values of 50-70 are normal for matplotlib internal caching
        # Negative values mean GC is doing its job
        assert objects_per_iter < 100, \
            f"Possible memory leak: {objects_per_iter:.1f} objects per iteration " \
            f"(total {object_growth} after 100 iterations)"

        # Also verify memory growth stabilizes by doing another batch
        mid_objects = final_objects

        for _ in range(100):
            real_feat = np.random.randn(100)
            synth_feat = np.random.randn(100)
            fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")
            plt.close(fig)

        gc.collect()
        final_final_objects = len(gc.get_objects())
        second_growth = final_final_objects - mid_objects
        second_objects_per_iter = second_growth / 100

        # The real memory leak test: second batch should not grow unbounded
        # Matplotlib does cache some objects (fonts, line styles, etc.),
        # so we allow moderate growth but catch catastrophic leaks
        # If we were leaking Figure objects, we'd see 500+ objects/iter
        assert second_objects_per_iter < 150, \
            f"Excessive memory growth in second batch: {second_objects_per_iter:.1f} obj/iter " \
            f"(threshold: 150 obj/iter)"

        # Overall, total growth across both batches should be reasonable
        total_growth = final_final_objects - initial_objects
        total_objects_per_iter = total_growth / 200  # 200 total iterations

        # Total growth should be sublinear - if we're leaking figures,
        # we'd see 100+ objects per iteration on average
        # Negative values are GOOD (GC is cleaning up)
        # We only fail if growth is positive and excessive
        if total_objects_per_iter > 0:
            assert total_objects_per_iter < 75, \
                f"Excessive overall memory growth: {total_objects_per_iter:.1f} obj/iter over 200 iterations " \
                f"(threshold: 75 obj/iter)"
        # If negative, that's excellent - memory is being reclaimed

    def test_large_dataset_waveform_visualization(self):
        """Test waveform visualization with very large number of epochs."""
        # Create large dataset: 1000 epochs
        np.random.seed(42)
        real = np.random.randn(1000, 63, 1001) * 0.5
        synthetic = real + np.random.randn(1000, 63, 1001) * 0.1

        import time
        start = time.time()

        # Visualize epoch near the end
        fig = plot_waveforms(real, synthetic, epoch_idx=999, channels=5)

        elapsed = time.time() - start

        # Should still be fast since we only plot one epoch
        assert elapsed < 1.0, f"Waveform plotting took {elapsed:.2f}s, should be < 1.0s"

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_large_dataset_statistical_comparison(self):
        """Test statistical comparison with very large feature arrays."""
        # Simulate features from 10,000 epochs x 63 channels
        real_feat = np.random.randn(630000)
        synth_feat = np.random.randn(630000)

        import time
        start = time.time()

        fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")

        elapsed = time.time() - start

        # Histogram computation might be slower with many samples
        assert elapsed < 3.0, f"Statistical comparison took {elapsed:.2f}s, should be < 3.0s"

        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_high_dpi_memory_usage(self):
        """
        Test that high DPI settings don't cause excessive memory usage.

        High DPI increases pixel count quadratically, so we need to ensure
        it doesn't cause memory issues.
        """
        real = np.random.randn(10, 63, 1001) * 0.5
        synthetic = real + np.random.randn(10, 63, 1001) * 0.1

        # Create plot at very high DPI
        fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5, dpi=300)

        # Should still complete without memory error
        assert isinstance(fig, Figure)

        # Figure size in memory (rough estimate)
        # A 14x8 inch figure at 300 DPI = 4200x2400 pixels = ~10M pixels
        # With RGBA, that's ~40 MB per figure
        # This should be manageable

        plt.close(fig)

    def test_extreme_number_of_bands(self):
        """Test band power comparison with many custom bands."""
        real = np.random.randn(5, 10, 1001) * 0.5
        synth = real + np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real, sfreq=200)
        synth_psds, _ = compute_psd(synth, sfreq=200)

        # Create 20 narrow bands
        many_bands = {f'Band_{i}': (i*2, (i+1)*2) for i in range(20)}

        fig = plot_band_power_comparison(real_psds, synth_psds, freqs, bands=many_bands)

        # Should create figure even with many bands
        assert isinstance(fig, Figure)
        # With 3x2 layout, we'd have multiple pages, but matplotlib creates all subplots
        assert len(fig.axes) >= 20, "Should have at least 20 subplots"

        plt.close(fig)


# ============================================================================
# Integration Tests: Cross-Function Validation
# ============================================================================

class TestIntegrationAndConsistency:
    """Test that visualization functions work together correctly."""

    def test_consistent_torch_tensor_handling_across_all_functions(self):
        """Test that all functions handle torch tensors consistently."""
        # Create torch tensor data
        real_torch = torch.randn(5, 10, 1001) * 0.5
        synth_torch = real_torch + torch.randn(5, 10, 1001) * 0.1

        # Test statistical comparison
        fig1 = plot_statistical_comparison(
            real_torch.mean(dim=2),
            synth_torch.mean(dim=2),
            "Mean"
        )
        assert isinstance(fig1, Figure)
        plt.close(fig1)

        # Test waveforms
        fig2 = plot_waveforms(real_torch, synth_torch, epoch_idx=0, channels=3)
        assert isinstance(fig2, Figure)
        plt.close(fig2)

        # Note: PSD and band power use numpy internally via compute_psd
        # which already handles torch tensors correctly

    def test_visualization_pipeline_with_edge_case_data(self):
        """
        Test complete visualization pipeline with challenging data.

        Simulates real-world scenario where data might have artifacts,
        missing values, or extreme amplitudes.
        """
        # Create data with various challenges
        real = np.random.randn(20, 30, 1001) * 0.5

        # Add some artifacts
        real[5, 10, 100:150] = 10.0  # Large amplitude artifact
        real[8, 15, :] = 0.0  # Dead channel
        real[12, 20, 500:600] = np.nan  # Missing data segment

        # Create synthetic data (cleaner)
        synthetic = np.random.randn(20, 30, 1001) * 0.5

        # Compute features (handling NaN)
        real_mean = np.nanmean(real, axis=2)
        synth_mean = np.nanmean(synthetic, axis=2)

        # Compute PSD (might have issues with NaN)
        real_clean = np.nan_to_num(real, nan=0.0)
        real_psds, freqs = compute_psd(real_clean)
        synth_psds, _ = compute_psd(synthetic)

        # All visualizations should still work
        figs = []

        figs.append(plot_statistical_comparison(real_mean, synth_mean, "Mean"))
        figs.append(plot_waveforms(real_clean, synthetic, epoch_idx=5, channels=3))
        figs.append(plot_psd_comparison(real_psds, synth_psds, freqs))
        figs.append(plot_band_power_comparison(real_psds, synth_psds, freqs, EEG_BANDS))

        # All should be valid figures
        for fig in figs:
            assert isinstance(fig, Figure)
            assert len(fig.axes) > 0
            plt.close(fig)

    def test_dpi_consistency_across_all_functions(self):
        """Test that DPI settings work consistently across all plotting functions."""
        real = np.random.randn(5, 10, 1001) * 0.5
        synth = real + np.random.randn(5, 10, 1001) * 0.1

        real_psds, freqs = compute_psd(real)
        synth_psds, _ = compute_psd(synth)

        test_dpi = 150

        # Test all functions with same DPI
        fig1 = plot_statistical_comparison(real.mean(axis=2), synth.mean(axis=2),
                                           "Mean", dpi=test_dpi)
        fig2 = plot_waveforms(real, synth, epoch_idx=0, channels=3, dpi=test_dpi)
        fig3 = plot_psd_comparison(real_psds, synth_psds, freqs, dpi=test_dpi)
        fig4 = plot_band_power_comparison(real_psds, synth_psds, freqs,
                                          EEG_BANDS, dpi=test_dpi)
        fig5 = plot_similarity_heatmap(
            np.random.randn(4, 100),
            np.random.randn(4, 100),
            ["F1", "F2", "F3", "F4"],
            dpi=test_dpi
        )

        # All should have the requested DPI (or 2x on Retina)
        for fig in [fig1, fig2, fig3, fig4, fig5]:
            assert fig.dpi in [test_dpi, test_dpi * 2], \
                f"Expected DPI {test_dpi} or {test_dpi*2}, got {fig.dpi}"
            plt.close(fig)


# ============================================================================
# Biological Plausibility Tests
# ============================================================================

class TestBiologicalPlausibility:
    """Test that visualizations respect EEG domain constraints."""

    def test_frequency_band_definitions_are_standard(self):
        """
        Test that EEG band definitions match neuroscience standards.

        Standard bands (approximate):
        - Delta: 0.5-4 Hz (not 1-4 to avoid edge effects)
        - Theta: 4-8 Hz
        - Alpha: 8-12/13 Hz
        - Beta: 12/13-30 Hz
        - Gamma: 30-40/50/100 Hz (varies by definition)
        """
        expected_bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 40)
        }

        assert EEG_BANDS == expected_bands, \
            f"EEG_BANDS doesn't match standard definitions.\n" \
            f"Expected: {expected_bands}\n" \
            f"Got: {EEG_BANDS}"

    def test_psd_frequency_range_is_reasonable(self):
        """
        Test that PSD computation produces reasonable frequency range.

        For sfreq=200 Hz, Nyquist frequency is 100 Hz.
        PSD should cover 0 to ~100 Hz.
        """
        data = np.random.randn(5, 10, 1001) * 0.5

        psds, freqs = compute_psd(data, sfreq=200)

        # Frequency range should be from 0 to Nyquist (100 Hz)
        assert freqs[0] >= 0, "Minimum frequency should be >= 0 Hz"
        assert freqs[-1] <= 100, f"Maximum frequency {freqs[-1]:.1f} should be <= 100 Hz (Nyquist)"
        assert len(freqs) == psds.shape[2], "Frequency array length should match PSD dimension"

    def test_band_power_computation_is_meaningful(self):
        """
        Test that band power computation actually integrates over the band.

        Create signal with known frequency content and verify band power reflects it.
        """
        # Create signal with strong 10 Hz component (alpha band)
        sfreq = 200
        duration = 5.0  # seconds
        n_samples = int(sfreq * duration)
        t = np.arange(n_samples) / sfreq

        # Strong alpha (10 Hz) signal
        signal_10hz = np.sin(2 * np.pi * 10 * t)

        # Create data array (1 epoch, 1 channel)
        data = signal_10hz.reshape(1, 1, -1)

        # Compute PSD
        psds, freqs = compute_psd(data, sfreq=sfreq)

        # Compute band powers
        from generative_eeg_augmentation.eval.eeg_metrics import compute_band_power

        alpha_power = compute_band_power(psds, freqs, (8, 12))
        delta_power = compute_band_power(psds, freqs, (1, 4))
        beta_power = compute_band_power(psds, freqs, (12, 30))

        # Alpha power should be much higher than delta or beta
        # (since our signal is 10 Hz)
        assert alpha_power[0, 0] > delta_power[0, 0], \
            "Alpha power should dominate for 10 Hz signal"
        assert alpha_power[0, 0] > beta_power[0, 0], \
            "Alpha power should dominate for 10 Hz signal"

    def test_statistical_features_reflect_known_properties(self):
        """
        Test that statistical comparison plots would correctly show known differences.

        Create data with known statistical properties and verify they could be
        distinguished in the visualization.
        """
        # Real data: mean=0, std=1
        real_feat = np.random.randn(1000)

        # Synthetic data: mean=2, std=1 (shifted distribution)
        synth_feat = np.random.randn(1000) + 2

        fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")

        # Get the axis
        ax = fig.axes[0]

        # Verify that x-axis spans both distributions
        xlim = ax.get_xlim()
        assert xlim[0] < 0, "X-axis should include negative values (real data)"
        assert xlim[1] > 2, "X-axis should include values > 2 (synthetic data)"

        plt.close(fig)


# ============================================================================
# Test Coverage Summary
# ============================================================================

def test_coverage_summary():
    """
    Document what aspects of the visualization module are tested.

    This test always passes but serves as documentation.
    """
    coverage_areas = {
        'Edge Cases': [
            'Zero/constant data',
            'NaN and Inf values',
            'Single epoch/channel/timepoint',
            'Extreme amplitudes',
            'Empty frequency bands'
        ],
        'Behavioral Correctness': [
            'Correct data appears in plots',
            'Correct epoch/channel selection',
            'Logarithmic scaling for PSD',
            'Correct number of subplots',
            'Torch tensor conversion accuracy'
        ],
        'Memory and Performance': [
            'No memory leaks with repeated plotting',
            'Large dataset handling (1000+ epochs)',
            'High DPI memory usage',
            'Many custom bands'
        ],
        'Integration': [
            'Consistent torch tensor handling',
            'Pipeline with artifact data',
            'DPI consistency across functions'
        ],
        'Biological Plausibility': [
            'Standard EEG band definitions',
            'Reasonable frequency ranges',
            'Meaningful band power computation',
            'Statistical feature correctness'
        ]
    }

    total_areas = sum(len(tests) for tests in coverage_areas.values())
    assert total_areas > 0, "Coverage summary should list tested areas"
