# Behavioral Testing Guide
## How to Write Tests That Actually Validate Correctness

This guide teaches you to write tests that catch real bugs, not just verify that code runs without crashing.

---

## The Golden Rule

**Your test should fail if the function returns wrong results, not just if it crashes.**

---

## Bad Test vs Good Test

### ❌ Bad: Structural Test (Too Superficial)

```python
def test_plot_waveforms():
    """Test that waveform plot returns a Figure."""
    fig = plot_waveforms(real, synth, epoch_idx=0, channels=5)
    assert isinstance(fig, Figure)  # Only checks type
    assert len(fig.axes) == 5        # Only checks structure
```

**Problem**: This test would pass even if:
- Plot shows wrong epoch
- Real and synthetic are swapped
- Plot is completely blank
- Legend is missing or incorrect

### ✅ Good: Behavioral Test (Verifies Correctness)

```python
def test_plot_waveforms_shows_correct_epoch():
    """Test that waveform plot actually plots the specified epoch."""
    real = np.random.randn(10, 63, 1001)
    synth = real * 0.8
    epoch_idx = 3

    fig = plot_waveforms(real, synth, epoch_idx=epoch_idx, channels=5)

    # CRITICAL: Extract actual plotted data
    line_data = fig.axes[0].get_lines()[0].get_ydata()

    # CRITICAL: Verify it matches expected input
    expected = real[epoch_idx, 0, :]
    assert np.allclose(line_data, expected, atol=1e-6), \
        "Plotted line doesn't match input data for specified epoch"
```

**Why Good**: This test extracts the actual plotted data and verifies it matches the input. It would catch:
- Wrong epoch being plotted
- Data corruption
- Incorrect indexing
- Scaling errors

---

## Testing Strategy by Function Type

### 1. Computation Functions (Metrics, Features)

**Test Strategy**: Verify against known inputs/outputs

```python
def test_compute_mean_manual_verification():
    """Verify mean computation matches manual calculation."""
    # Known input with known output
    data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])  # 1 epoch, 1 channel

    mean, _, _, _ = compute_time_domain_features(data)

    # Manual calculation: (1+2+3+4+5)/5 = 3.0
    expected_mean = 3.0
    assert abs(mean[0, 0] - expected_mean) < 0.001, \
        f"Mean calculation incorrect: got {mean[0,0]}, expected {expected_mean}"
```

**Key Principle**: Create inputs where you know the correct output.

### 2. Signal Processing Functions (PSD, Filtering)

**Test Strategy**: Use synthetic signals with known properties

```python
def test_psd_detects_10hz_sine_wave():
    """Verify PSD correctly identifies a 10Hz sine wave."""
    sfreq = 200
    duration = 5
    t = np.linspace(0, duration, sfreq * duration)

    # Create pure 10Hz sine wave
    signal = np.sin(2 * np.pi * 10 * t)
    data = signal.reshape(1, 1, -1)

    psds, freqs = compute_psd(data, sfreq=sfreq)

    # Find peak frequency
    peak_idx = np.argmax(psds[0, 0, :])
    peak_freq = freqs[peak_idx]

    # Peak should be at 10Hz (±1Hz tolerance for windowing)
    assert abs(peak_freq - 10.0) < 1.0, \
        f"PSD peak at {peak_freq:.2f}Hz, expected 10Hz"
```

**Key Principle**: Create signals with known spectral properties and verify detection.

### 3. Visualization Functions (Plots)

**Test Strategy**: Extract rendered data and verify against input

```python
def test_plot_shows_correct_data():
    """Verify plot actually visualizes the input data."""
    real_psd = np.random.rand(1, 1, 100)
    freqs = np.linspace(0, 50, 100)

    fig = plot_psd_comparison(real_psd, synth_psd, freqs, epoch_idx=0, ch_idx=0)

    # Extract plotted line data
    ax = fig.axes[0]
    line_x = ax.get_lines()[0].get_xdata()
    line_y = ax.get_lines()[0].get_ydata()

    # Verify X-axis matches frequency array
    assert np.allclose(line_x, freqs, atol=1e-6), \
        "Plotted X-axis doesn't match frequency array"

    # Verify Y-axis matches PSD values
    assert np.allclose(line_y, real_psd[0, 0, :], atol=1e-6), \
        "Plotted Y-axis doesn't match input PSD"
```

**Key Principle**: Don't trust that it "looks right" - extract and verify.

### 4. Data Loading Functions

**Test Strategy**: Verify data properties, not just shapes

```python
def test_data_has_temporal_autocorrelation():
    """Verify loaded EEG has temporal structure (not white noise)."""
    data, labels = load_demo_preprocessed()

    # Get first epoch, first channel
    signal = data[0, 0, :]

    # Compute lag-1 autocorrelation
    autocorr = np.corrcoef(signal[:-1], signal[1:])[0, 1]

    # Real EEG has positive autocorrelation (adjacent samples similar)
    # White noise has autocorr ≈ 0
    assert autocorr > 0.3, \
        f"Data has low autocorrelation ({autocorr:.4f}), may be corrupted"
```

**Key Principle**: Test that data has expected statistical/physical properties.

---

## Categories of Behavioral Tests

### 1. Known Input/Output Tests

Test with inputs where you know the exact expected output.

```python
def test_variance_of_constant_signal_is_zero():
    """Constant signal must have zero variance."""
    data = np.ones((5, 10, 100)) * 7.0  # Constant value 7.0

    _, var, _, _ = compute_time_domain_features(data)

    assert np.allclose(var, 0.0, atol=1e-10), \
        "Constant signal should have zero variance"
```

### 2. Physical Constraint Tests

Test mathematical/physical laws that must hold.

```python
def test_variance_always_nonnegative():
    """Variance cannot be negative (it's a squared quantity)."""
    data = np.random.randn(20, 63, 1001) * 50

    _, var, _, _ = compute_time_domain_features(data)

    assert np.all(var >= 0), \
        f"Variance must be non-negative, found min={np.min(var)}"
```

### 3. Consistency Tests

Test that related quantities are consistent.

```python
def test_psd_power_matches_time_domain_variance():
    """Verify Parseval's theorem: time variance ≈ frequency power."""
    data = np.random.randn(5, 10, 1000)

    # Time domain variance
    time_var = np.var(data, axis=2)

    # Frequency domain power
    psds, freqs = compute_psd(data, sfreq=200)
    freq_df = freqs[1] - freqs[0]
    freq_power = np.sum(psds, axis=2) * freq_df

    # Should match within 50% (windowing causes some loss)
    ratio = freq_power / (time_var + 1e-10)
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), \
        "PSD power doesn't match time variance (Parseval's theorem)"
```

### 4. Sensitivity Tests

Test that function responds appropriately to changes.

```python
def test_band_power_responds_to_frequency():
    """Alpha band should respond to 10Hz, not 20Hz."""
    sfreq = 200
    t = np.linspace(0, 5, sfreq * 5)

    # 10Hz signal (in Alpha band)
    alpha_signal = np.sin(2 * np.pi * 10 * t).reshape(1, 1, -1)
    # 20Hz signal (in Beta band)
    beta_signal = np.sin(2 * np.pi * 20 * t).reshape(1, 1, -1)

    alpha_psds, freqs = compute_psd(alpha_signal, sfreq=sfreq)
    beta_psds, _ = compute_psd(beta_signal, sfreq=sfreq)

    alpha_powers = compute_all_band_powers(alpha_psds, freqs)
    beta_powers = compute_all_band_powers(beta_psds, freqs)

    # For 10Hz signal: Alpha power >> Beta power
    assert alpha_powers['Alpha'] > alpha_powers['Beta'] * 10, \
        "10Hz signal should have much more Alpha than Beta power"

    # For 20Hz signal: Beta power >> Alpha power
    assert beta_powers['Beta'] > beta_powers['Alpha'] * 10, \
        "20Hz signal should have much more Beta than Alpha power"
```

---

## Common Pitfalls to Avoid

### Pitfall 1: Testing Only Happy Path

❌ **Bad**:
```python
def test_load_data():
    data, labels = load_all_preprocessed()
    assert data.shape[0] > 0
```

✅ **Good**:
```python
def test_load_data_happy_path():
    data, labels = load_all_preprocessed()
    assert data.shape[0] > 0

def test_load_data_missing_directory():
    with pytest.raises(RuntimeError, match="not found"):
        load_all_preprocessed(preprocessed_root="/nonexistent")

def test_load_data_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError, match="No preprocessed data"):
        load_all_preprocessed(preprocessed_root=empty_dir)
```

### Pitfall 2: Testing Implementation Details

❌ **Bad** (tests internal variable names):
```python
def test_generator_has_fc1_layer():
    gen = EEGGenerator()
    assert hasattr(gen, 'fc1')  # Internal implementation detail
```

✅ **Good** (tests behavior):
```python
def test_generator_produces_valid_output():
    gen = EEGGenerator()
    noise = torch.randn(10, 100)
    labels = torch.zeros(10, 2)
    labels[:, 0] = 1

    output = gen(noise, labels)

    assert output.shape == (10, 63, 1001), "Wrong output shape"
    assert output.min() >= -1.0 and output.max() <= 1.0, \
        "Output should be in [-1, 1] range (tanh activation)"
```

### Pitfall 3: Assertions Without Messages

❌ **Bad**:
```python
assert len(fig.axes) == 5
assert autocorr > 0.3
```

✅ **Good**:
```python
assert len(fig.axes) == 5, \
    f"Expected 5 subplots for 5 channels, got {len(fig.axes)}"

assert autocorr > 0.3, \
    f"Data has low autocorrelation ({autocorr:.4f}). " \
    "May be corrupted or white noise."
```

**Why**: When test fails, message explains what went wrong and why it matters.

---

## Quick Reference: Test Quality Checklist

Before committing a test, ask:

- [ ] Would this test fail if I returned wrong values (not just if I crashed)?
- [ ] Does this test verify actual correctness, not just structure?
- [ ] Does the assertion message explain what went wrong?
- [ ] Does the test check a specific behavior, not implementation details?
- [ ] Have I tested edge cases (empty input, boundary values)?
- [ ] Does the test have a clear docstring explaining what it validates?

---

## Examples of Excellent Tests from This Project

### Example 1: Frequency Detection (from test_metrics.py)
```python
def test_psd_frequency_detection(self):
    """Verify PSD correctly identifies frequency peaks. (EXCELLENT)"""
    sfreq = 200
    duration = 5
    t = np.linspace(0, duration, sfreq * duration)
    signal = np.sin(2 * np.pi * 10 * t)  # 10Hz sine
    data = signal.reshape(1, 1, -1)

    psds, freqs = compute_psd(data, sfreq=sfreq)
    peak_idx = np.argmax(psds[0, 0, :])
    peak_freq = freqs[peak_idx]

    assert abs(peak_freq - 10.0) < 1.0, \
        f"PSD peak at {peak_freq:.2f}Hz, expected 10Hz"
```
**Why Excellent**: Creates known signal, verifies detection works.

### Example 2: Temporal Structure (from test_data_loader.py)
```python
def test_data_loader_temporal_structure(self):
    """Verify loaded data has temporal structure. (EXCELLENT)"""
    data, labels = load_demo_preprocessed()
    epoch0 = data[0, 0, :]
    autocorr = np.corrcoef(epoch0[:-1], epoch0[1:])[0, 1]

    assert autocorr > 0.3, \
        f"Data has low autocorrelation ({autocorr:.4f}). May be corrupted."
```
**Why Excellent**: Tests that data has expected physical property (EEG autocorrelation).

### Example 3: Plot Content Validation (from new tests)
```python
def test_plots_correct_psd_data(self):
    """Verify plot shows correct PSD values. (EXCELLENT)"""
    real_psds, freqs = compute_psd(data, sfreq=200)
    fig = plot_psd_comparison(real_psds, synth_psds, freqs,
                              epoch_idx=2, ch_idx=5)

    # Extract plotted data
    line_y = fig.axes[0].get_lines()[0].get_ydata()
    expected = real_psds[2, 5, :]

    assert np.allclose(line_y, expected, atol=1e-6), \
        "Plotted PSD doesn't match input for epoch 2, channel 5"
```
**Why Excellent**: Extracts actual rendered data and verifies against input.

---

## When to Write Behavioral Tests

**Always** write behavioral tests for:
- Functions that compute metrics or features
- Functions that process signals (filtering, PSD, etc.)
- Functions that load or transform data
- Functions that generate visualizations
- Functions that make decisions based on data

**Structural tests are OK** for:
- Constructor parameter passing
- Type conversions (torch ↔ numpy)
- File existence checks
- Basic shape validation

**But never rely on structural tests alone!**

---

## Testing Workflow

1. **Write the function**
2. **Write structural test** (does it run without crashing?)
3. **Write behavioral test** (does it produce correct results?)
4. **Write edge case tests** (what if input is weird?)
5. **Write error handling tests** (what if input is invalid?)

If you only do step 2, you're testing that your code runs, not that it works.

---

## Further Reading

- Original tests: `tests/test_metrics.py` (excellent examples)
- New behavioral tests: `tests/test_visualizations_behavioral.py`
- Assessment report: `TEST_QUALITY_ASSESSMENT.md`
- This project's philosophy: "Does it work?" > "Does it crash?"

---

**Remember**: A test that always passes is worse than no test at all, because it gives false confidence.
