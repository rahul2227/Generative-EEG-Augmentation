# Behavioral Testing Examples from Milestone 4

This document showcases how the visualization module tests follow behavioral testing principles, demonstrating the difference between structural tests (does it crash?) and behavioral tests (does it work correctly?).

## Principle 1: Test Correctness, Not Just Absence of Crashes

### ❌ Structural Test (Insufficient)
```python
def test_plot_waveforms_works():
    """Test that plot_waveforms doesn't crash."""
    real = np.random.randn(10, 63, 1001)
    synthetic = np.random.randn(10, 63, 1001)

    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=5)

    assert isinstance(fig, Figure)  # Only tests it returns a figure
```

**Problem**: This test would pass even if the function plotted the wrong epoch, wrong channels, or random data!

### ✅ Behavioral Test (Correct)
```python
def test_waveforms_plot_correct_epoch():
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
    real_line_data = lines[0].get_ydata()

    # Verify it's approximately epoch 3's data (mean ~30)
    assert np.abs(np.mean(real_line_data) - 30) < 5, \
        f"Plotted data mean {np.mean(real_line_data):.2f} should be near 30"

    plt.close(fig)
```

**Why Better**: This test would FAIL if the function plotted the wrong epoch, even though it wouldn't crash.

---

## Principle 2: Validate Domain-Specific Properties

### ❌ Structural Test (Insufficient)
```python
def test_psd_comparison_creates_plot():
    """Test that PSD comparison creates a plot."""
    real_psds = np.random.rand(10, 63, 129)
    synth_psds = np.random.rand(10, 63, 129)
    freqs = np.linspace(0, 100, 129)

    fig = plot_psd_comparison(real_psds, synth_psds, freqs)

    assert len(fig.axes) > 0  # Only tests it has axes
```

**Problem**: Doesn't verify the critical EEG requirement that PSD uses logarithmic scale!

### ✅ Behavioral Test (Correct)
```python
def test_psd_comparison_uses_semilogy_scale():
    """
    Test that PSD comparison uses logarithmic y-axis.

    This is critical for EEG PSD visualization as power varies
    over orders of magnitude.
    """
    real = np.random.randn(5, 10, 1001) * 0.5
    synth = real + np.random.randn(5, 10, 1001) * 0.1

    real_psds, freqs = compute_psd(real)
    synth_psds, _ = compute_psd(synth)

    fig = plot_psd_comparison(real_psds, synth_psds, freqs)

    ax = fig.axes[0]

    # Verify y-axis uses log scale (critical domain requirement)
    assert ax.get_yscale() == 'log', \
        "PSD plot should use logarithmic y-axis"

    plt.close(fig)
```

**Why Better**: This test enforces a biological/domain requirement that is critical for correct EEG visualization.

---

## Principle 3: Test Known Input → Known Output

### ❌ Structural Test (Insufficient)
```python
def test_band_power_computation():
    """Test that band power computation works."""
    psds = np.random.rand(10, 63, 129)
    freqs = np.linspace(0, 100, 129)

    alpha_power = compute_band_power(psds, freqs, (8, 12))

    assert alpha_power.shape == (10, 63)  # Only tests shape
```

**Problem**: This test would pass even if the function returned all zeros or random numbers!

### ✅ Behavioral Test (Correct)
```python
def test_band_power_computation_is_meaningful():
    """
    Test that band power computation actually integrates over the band.

    Create signal with known frequency content and verify band power
    reflects it.
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
    alpha_power = compute_band_power(psds, freqs, (8, 12))
    delta_power = compute_band_power(psds, freqs, (1, 4))
    beta_power = compute_band_power(psds, freqs, (12, 30))

    # Alpha power should be much higher than delta or beta
    # (since our signal is 10 Hz)
    assert alpha_power[0, 0] > delta_power[0, 0], \
        "Alpha power should dominate for 10 Hz signal"
    assert alpha_power[0, 0] > beta_power[0, 0], \
        "Alpha power should dominate for 10 Hz signal"
```

**Why Better**: Uses a signal with KNOWN frequency content (10 Hz) and verifies the function correctly identifies high power in the alpha band. Would fail if computation was wrong.

---

## Principle 4: Test Edge Cases for Correctness, Not Just Survival

### ❌ Structural Test (Insufficient)
```python
def test_plot_with_single_channel():
    """Test that plotting works with single channel."""
    real = np.random.randn(10, 1, 1001)
    synthetic = np.random.randn(10, 1, 1001)

    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)

    assert fig is not None  # Only tests it doesn't crash
```

**Problem**: Doesn't verify that the plot is actually correct for the edge case!

### ✅ Behavioral Test (Correct)
```python
def test_plot_waveforms_with_single_channel():
    """Test waveform plotting with single channel data."""
    real = np.random.randn(10, 1, 1001) * 0.5
    synthetic = real + np.random.randn(10, 1, 1001) * 0.1

    fig = plot_waveforms(real, synthetic, epoch_idx=0, channels=1)

    # Verify correct behavior for edge case
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1, \
        "Should have exactly 1 subplot for 1 channel"

    # Verify the subplot actually contains data
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) >= 2, \
        "Should have at least 2 lines (real and synthetic)"

    plt.close(fig)
```

**Why Better**: Verifies that the edge case produces the correct structure (1 subplot) and contains data (2 lines).

---

## Principle 5: Validate Data Appears in the Visualization

### ❌ Structural Test (Insufficient)
```python
def test_statistical_comparison():
    """Test statistical comparison plot."""
    real_feat = np.random.randn(100)
    synth_feat = np.random.randn(100)

    fig = plot_statistical_comparison(real_feat, synth_feat, "Mean")

    assert len(fig.axes) > 0  # Only tests plot exists
```

**Problem**: Doesn't verify that the input data is actually displayed!

### ✅ Behavioral Test (Correct)
```python
def test_statistical_comparison_data_appears_in_plot():
    """
    Test that input data actually appears in the statistical
    comparison plot.

    Validates that the plot contains the actual data values,
    not arbitrary numbers.
    """
    # Create distinctive distributions
    real_feat = np.random.randn(100) * 2 + 10   # Mean ~10, std ~2
    synth_feat = np.random.randn(100) * 2 - 10  # Mean ~-10, std ~2

    fig = plot_statistical_comparison(real_feat, synth_feat, "TestFeature")

    # Extract data from the plot
    ax = fig.axes[0]

    # The plot should have created histogram patches
    assert len(ax.patches) > 0, "Plot should contain histogram bars"

    # Check that x-axis range encompasses the data
    xlim = ax.get_xlim()
    assert xlim[0] < real_feat.min(), \
        "X-axis should cover minimum value"
    assert xlim[1] > synth_feat.max(), \
        "X-axis should cover maximum value"

    plt.close(fig)
```

**Why Better**: Verifies that the plot's x-axis range actually encompasses the input data. Would fail if the function plotted wrong data or hardcoded ranges.

---

## Principle 6: Test Biological Plausibility

### ❌ Generic Test (Misses Domain Knowledge)
```python
def test_eeg_bands_exist():
    """Test that EEG bands are defined."""
    assert len(EEG_BANDS) > 0  # Only tests dictionary isn't empty
```

**Problem**: Doesn't verify bands match neuroscience standards!

### ✅ Domain-Aware Test (Correct)
```python
def test_frequency_band_definitions_are_standard():
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
```

**Why Better**: Enforces that the code uses scientifically correct frequency band definitions. Would catch if someone accidentally changed Delta to (1, 10) Hz.

---

## Principle 7: Test for Memory Efficiency, Not Just No Crashes

### ❌ Structural Test (Insufficient)
```python
def test_repeated_plotting():
    """Test that we can create multiple plots."""
    for _ in range(10):
        real = np.random.randn(100)
        synth = np.random.randn(100)
        fig = plot_statistical_comparison(real, synth, "Mean")
        plt.close(fig)

    # No assertion - just checking it doesn't crash
```

**Problem**: Doesn't detect memory leaks!

### ✅ Behavioral Test (Correct)
```python
def test_memory_leak_repeated_plotting():
    """
    Test that repeated plot generation doesn't cause unbounded
    memory growth.

    This is critical for applications that generate many plots
    (e.g., Streamlit apps).
    """
    import gc

    # Warmup: establish baseline caches
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

    objects_per_iter = (final_objects - initial_objects) / 100

    # If we're leaking Figure objects, we'd see 500+ objects/iter
    # Matplotlib caching creates ~50-70 objects/iter (normal)
    assert objects_per_iter < 100, \
        f"Possible memory leak: {objects_per_iter:.1f} objects per iteration"
```

**Why Better**: Actually measures memory usage and detects leaks. Would fail if `plt.close()` wasn't being called or figures weren't being released.

---

## Summary: Behavioral vs Structural Testing

| Aspect | Structural Test | Behavioral Test |
|--------|----------------|-----------------|
| **Focus** | Doesn't crash | Produces correct results |
| **Validation** | Type/shape checking | Data correctness |
| **Domain Knowledge** | Generic assertions | EEG-specific requirements |
| **Edge Cases** | Tests survival | Tests correctness |
| **False Positives** | Would pass buggy code | Catches subtle bugs |
| **Maintenance** | Low value | High value |

## Key Takeaways

1. **Always ask "Would this test fail if the code was wrong?"**
   - If a buggy implementation would still pass, the test is insufficient

2. **Use domain knowledge in tests**
   - EEG requires log-scale PSD plots
   - Alpha band should contain power from 10 Hz signals
   - Band definitions must match neuroscience standards

3. **Create known inputs with expected outputs**
   - 10 Hz signal should show alpha band power
   - Epoch 3 with mean 30 should plot data with mean ~30
   - Identical real/synthetic should show high correlation

4. **Validate the actual content of visualizations**
   - Extract axis limits and compare to data ranges
   - Check line data matches input data
   - Verify scale types (linear vs log)

5. **Test for efficiency, not just functionality**
   - Memory leaks over many iterations
   - Performance with large datasets
   - Stability of resource usage

---

**The Gold Standard**: A test should fail when the implementation produces incorrect results, even if it doesn't crash. If you can make a test pass with obviously wrong code, it's not a good test.
