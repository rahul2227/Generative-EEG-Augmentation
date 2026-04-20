"""
Additional behavioral tests for data loader edge cases.

These tests focus on scenarios that weren't fully covered in the original
test suite, particularly edge cases around single event codes, missing data,
and parameter validation.
"""

import pytest
import numpy as np
import mne
from pathlib import Path
from unittest.mock import Mock, patch
from generative_eeg_augmentation.data.loader import (
    load_demo_preprocessed,
    load_all_preprocessed
)


class TestLoadAllPreprocessedEdgeCases:
    """Test edge cases for load_all_preprocessed function."""

    def test_handles_nonexistent_directory_gracefully(self):
        """
        Test that nonexistent directory raises informative error.

        CRITICAL: Tests error handling for missing data directory.
        """
        nonexistent_path = Path("/nonexistent/preprocessed/data")

        with pytest.raises(RuntimeError, match="Preprocessed data directory not found"):
            load_all_preprocessed(preprocessed_root=nonexistent_path)

    def test_handles_empty_directory_gracefully(self, tmp_path):
        """
        Test that empty directory (no subjects) raises informative error.

        CRITICAL: Tests error handling when no data files exist.
        """
        empty_dir = tmp_path / "empty_preprocessed"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError, match="No preprocessed data found"):
            load_all_preprocessed(preprocessed_root=empty_dir)

    def test_skips_subjects_without_fif_files(self, tmp_path):
        """
        Test that subjects without preprocessed_epochs-epo.fif are skipped.

        CRITICAL: Tests graceful handling of incomplete subject data.
        """
        preprocessed_root = tmp_path / "preprocessed"
        preprocessed_root.mkdir()

        # Create subject folder WITHOUT .fif file
        subject_no_fif = preprocessed_root / "S001"
        subject_no_fif.mkdir()
        (subject_no_fif / "other_file.txt").write_text("dummy")

        # Create another subject WITH .fif file (to avoid "no data" error)
        subject_with_fif = preprocessed_root / "S002"
        subject_with_fif.mkdir()

        # Create minimal valid epochs object
        info = mne.create_info(['Ch1', 'Ch2'], sfreq=200, ch_types='eeg')
        data = np.random.randn(5, 2, 100)  # 5 epochs, 2 channels, 100 timepoints
        events = np.array([[i, 0, 1] for i in range(5)])  # All same event code
        epochs = mne.EpochsArray(data, info, events=events, tmin=0)
        fif_path = subject_with_fif / "preprocessed_epochs-epo.fif"
        epochs.save(fif_path, overwrite=True, verbose=False)

        # Should not crash, should skip S001 and load S002
        data_loaded, labels_loaded = load_all_preprocessed(preprocessed_root=preprocessed_root)

        # Verify data was loaded (only from S002)
        assert data_loaded.shape[0] == 5, "Should load 5 epochs from S002"
        assert data_loaded.shape[1] == 2, "Should have 2 channels"

    def test_one_hot_encoding_sums_to_one(self):
        """
        Test that one-hot encoded labels sum to 1.0 along class dimension.

        CRITICAL: Verifies correctness of one-hot encoding (not just shape).
        """
        data, labels = load_demo_preprocessed()

        # Sum across class dimension (axis=1) should equal 1.0 for each sample
        label_sums = labels.sum(axis=1)

        assert np.allclose(label_sums, 1.0, atol=1e-6), \
            "One-hot encoded labels should sum to 1.0 for each sample"

    def test_labels_are_binary_values(self):
        """
        Test that label values are strictly 0 or 1 (no intermediate values).

        CRITICAL: Verifies one-hot encoding uses binary values, not probabilities.
        """
        data, labels = load_demo_preprocessed()

        # All label values should be either 0.0 or 1.0
        unique_values = np.unique(labels)
        assert len(unique_values) <= 2, "Labels should have at most 2 unique values (0 and 1)"
        assert np.all(np.isin(unique_values, [0.0, 1.0])), \
            "Label values should be exactly 0.0 or 1.0"

    def test_num_classes_parameter_respected(self, tmp_path):
        """
        Test that num_classes parameter controls output shape.

        CRITICAL: Verifies parameter is actually used in one-hot encoding.
        """
        preprocessed_root = tmp_path / "preprocessed"
        preprocessed_root.mkdir()
        subject = preprocessed_root / "S001"
        subject.mkdir()

        # Create epochs with 2 event codes
        info = mne.create_info(['Ch1'], sfreq=200, ch_types='eeg')
        data = np.random.randn(10, 1, 100)
        events = np.array([[i, 0, i % 2 + 1] for i in range(10)])  # Event codes 1 and 2
        epochs = mne.EpochsArray(data, info, events=events, tmin=0)
        fif_path = subject / "preprocessed_epochs-epo.fif"
        epochs.save(fif_path, overwrite=True, verbose=False)

        # Load with num_classes=2 (default)
        data_2class, labels_2class = load_all_preprocessed(
            preprocessed_root=preprocessed_root, num_classes=2
        )
        assert labels_2class.shape[1] == 2, "Should have 2 classes with num_classes=2"

        # Load with num_classes=3
        data_3class, labels_3class = load_all_preprocessed(
            preprocessed_root=preprocessed_root, num_classes=3
        )
        assert labels_3class.shape[1] == 3, "Should have 3 classes with num_classes=3"

    def test_handles_single_event_code_with_random_split(self, tmp_path):
        """
        Test behavior when subject has only one event code (triggers random split).

        CRITICAL: Verifies special case handling in lines 68-75 of loader.py.
        """
        preprocessed_root = tmp_path / "preprocessed"
        preprocessed_root.mkdir()
        subject = preprocessed_root / "S001"
        subject.mkdir()

        # Create epochs with ONLY ONE event code
        info = mne.create_info(['Ch1', 'Ch2'], sfreq=200, ch_types='eeg')
        n_epochs = 20
        data = np.random.randn(n_epochs, 2, 100)
        events = np.array([[i, 0, 1] for i in range(n_epochs)])  # All event code 1
        epochs = mne.EpochsArray(data, info, events=events, tmin=0)
        fif_path = subject / "preprocessed_epochs-epo.fif"
        epochs.save(fif_path, overwrite=True, verbose=False)

        # Load data (should trigger random split)
        data_loaded, labels_loaded = load_all_preprocessed(preprocessed_root=preprocessed_root)

        # Verify both classes have samples (approximately half each)
        class0_count = (labels_loaded[:, 0] == 1).sum()
        class1_count = (labels_loaded[:, 1] == 1).sum()

        assert class0_count > 0, "Class 0 should have samples after random split"
        assert class1_count > 0, "Class 1 should have samples after random split"
        assert class0_count + class1_count == n_epochs, "All epochs should be labeled"

        # Split should be roughly balanced (within 30% of 50/50)
        split_ratio = min(class0_count, class1_count) / max(class0_count, class1_count)
        assert split_ratio > 0.3, \
            f"Random split should be roughly balanced, got {class0_count}:{class1_count}"

    def test_handles_more_than_two_event_codes(self, tmp_path):
        """
        Test that loader takes only first two event codes when more than two exist.

        CRITICAL: Verifies behavior documented in lines 78-80 of loader.py.
        """
        preprocessed_root = tmp_path / "preprocessed"
        preprocessed_root.mkdir()
        subject = preprocessed_root / "S001"
        subject.mkdir()

        # Create epochs with THREE event codes
        info = mne.create_info(['Ch1'], sfreq=200, ch_types='eeg')
        n_epochs = 30
        data = np.random.randn(n_epochs, 1, 100)
        # Event codes: 10, 20, 30 (cycling)
        events = np.array([[i, 0, 10 + (i % 3) * 10] for i in range(n_epochs)])
        epochs = mne.EpochsArray(data, info, events=events, tmin=0)
        fif_path = subject / "preprocessed_epochs-epo.fif"
        epochs.save(fif_path, overwrite=True, verbose=False)

        # Load data (should use only first two unique codes: 10 and 20)
        data_loaded, labels_loaded = load_all_preprocessed(preprocessed_root=preprocessed_root)

        # All epochs should be labeled (mapped to class 0 or 1)
        assert labels_loaded.shape[0] == n_epochs, "All epochs should be loaded"

        # Both classes should have samples
        class0_count = (labels_loaded[:, 0] == 1).sum()
        class1_count = (labels_loaded[:, 1] == 1).sum()

        assert class0_count > 0, "Class 0 should have samples"
        assert class1_count > 0, "Class 1 should have samples"
        assert class0_count + class1_count == n_epochs, "All epochs should be classified"

    def test_data_shape_consistency_across_subjects(self, tmp_path):
        """
        Test that data from multiple subjects is concatenated correctly.

        CRITICAL: Verifies concatenation maintains consistent shapes.
        """
        preprocessed_root = tmp_path / "preprocessed"
        preprocessed_root.mkdir()

        # Create two subjects with different numbers of epochs
        for subj_idx, n_epochs in enumerate([10, 15], start=1):
            subject = preprocessed_root / f"S00{subj_idx}"
            subject.mkdir()

            info = mne.create_info(['Ch1', 'Ch2', 'Ch3'], sfreq=200, ch_types='eeg')
            data = np.random.randn(n_epochs, 3, 100)
            events = np.array([[i, 0, i % 2 + 1] for i in range(n_epochs)])
            epochs = mne.EpochsArray(data, info, events=events, tmin=0)
            fif_path = subject / "preprocessed_epochs-epo.fif"
            epochs.save(fif_path, overwrite=True, verbose=False)

        # Load all subjects
        data_loaded, labels_loaded = load_all_preprocessed(preprocessed_root=preprocessed_root)

        # Total epochs should be sum of both subjects
        assert data_loaded.shape[0] == 10 + 15, "Should concatenate epochs from both subjects"

        # Channel and timepoint dimensions should match
        assert data_loaded.shape[1] == 3, "Should have 3 channels from subject data"
        assert data_loaded.shape[2] == 100, "Should have 100 timepoints from subject data"

        # Labels should match epoch count
        assert labels_loaded.shape[0] == 25, "Labels should match total epochs"


class TestLoadDemoPreprocessedEdgeCases:
    """Test edge cases for load_demo_preprocessed function."""

    def test_demo_data_channel_count_matches_full_dataset(self):
        """
        Test that demo data has same channel count as full dataset.

        CRITICAL: Verifies demo is a true subset (compatible shapes).
        """
        demo_data, demo_labels = load_demo_preprocessed()
        full_data, full_labels = load_all_preprocessed()

        assert demo_data.shape[1] == full_data.shape[1], \
            "Demo and full data should have same number of channels"
        assert demo_data.shape[2] == full_data.shape[2], \
            "Demo and full data should have same number of timepoints"

    def test_demo_data_smaller_than_full_dataset(self):
        """
        Test that demo dataset is actually smaller than full dataset.

        CRITICAL: Verifies demo serves its purpose (quick loading).
        """
        demo_data, demo_labels = load_demo_preprocessed()
        full_data, full_labels = load_all_preprocessed()

        assert demo_data.shape[0] < full_data.shape[0], \
            "Demo dataset should have fewer epochs than full dataset"

        # Demo should be significantly smaller (< 20% of full)
        ratio = demo_data.shape[0] / full_data.shape[0]
        assert ratio < 0.2, \
            f"Demo should be much smaller than full dataset (got {ratio:.2%})"

    def test_demo_labels_balanced_across_classes(self):
        """
        Test that demo dataset has relatively balanced classes.

        CRITICAL: Verifies demo is useful for testing (both classes present).
        """
        demo_data, demo_labels = load_demo_preprocessed()

        class0_count = (demo_labels[:, 0] == 1).sum()
        class1_count = (demo_labels[:, 1] == 1).sum()

        # Both classes should be present
        assert class0_count > 0, "Demo should have class 0 samples"
        assert class1_count > 0, "Demo should have class 1 samples"

        # Should be reasonably balanced (20-80% split is acceptable)
        ratio = min(class0_count, class1_count) / max(class0_count, class1_count)
        assert ratio > 0.2, \
            f"Demo classes should be reasonably balanced, got {class0_count}:{class1_count}"


class TestDataLoaderNumericalStability:
    """Test numerical stability and value correctness."""

    def test_no_nan_or_inf_in_data(self):
        """
        Test that loaded data contains no NaN or Inf values.

        CRITICAL: Verifies data quality for downstream processing.
        """
        data, labels = load_demo_preprocessed()

        assert not np.any(np.isnan(data)), "Data should not contain NaN values"
        assert not np.any(np.isinf(data)), "Data should not contain Inf values"
        assert not np.any(np.isnan(labels)), "Labels should not contain NaN values"
        assert not np.any(np.isinf(labels)), "Labels should not contain Inf values"

    def test_data_dtype_is_floating_point(self):
        """
        Test that data has floating-point dtype (required for ML).

        CRITICAL: Verifies data can be used in neural network training.
        """
        data, labels = load_demo_preprocessed()

        assert np.issubdtype(data.dtype, np.floating), \
            f"Data should be floating point, got {data.dtype}"
        assert np.issubdtype(labels.dtype, np.floating), \
            f"Labels should be floating point, got {labels.dtype}"

    def test_data_mean_approximately_zero(self):
        """
        Test that preprocessed data has near-zero mean (preprocessing check).

        CRITICAL: Verifies preprocessing was applied correctly.
        """
        data, labels = load_demo_preprocessed()

        overall_mean = np.mean(data)

        # Preprocessed EEG should be centered (mean near 0)
        # Allow tolerance for biological variation and preprocessing artifacts
        assert abs(overall_mean) < 20, \
            f"Preprocessed data should have near-zero mean, got {overall_mean:.2f}"

    def test_data_variance_reasonable_for_eeg(self):
        """
        Test that data variance is in expected range for EEG.

        CRITICAL: Verifies data scaling is appropriate.
        """
        data, labels = load_demo_preprocessed()

        overall_std = np.std(data)

        # EEG standard deviation:
        # - Raw EEG: typically 10-100 μV (std ≈ 20-50)
        # - Standardized EEG: std ≈ 1
        # Accept both ranges
        assert (0.5 < overall_std < 2.0) or (5 < overall_std < 200), \
            f"Data std ({overall_std:.2f}) outside expected ranges for EEG"

    def test_epoch_length_matches_sampling_rate_convention(self):
        """
        Test that epoch length is consistent with sampling rate conventions.

        CRITICAL: Verifies temporal properties are correct.
        """
        data, labels = load_demo_preprocessed()

        n_timepoints = data.shape[2]

        # Common epoch lengths at 200 Hz sampling rate:
        # - 1 second = 200 samples
        # - 2 seconds = 400 samples
        # - 5 seconds = 1000 samples
        # Current data has 1001 samples (5 seconds + 1)

        # Verify timepoints is a reasonable epoch length
        assert n_timepoints > 100, "Epoch should have at least 0.5 seconds of data (100 samples at 200Hz)"
        assert n_timepoints < 5000, "Epoch should be less than 25 seconds (5000 samples at 200Hz)"

        # Verify it's close to a round number (within 10 samples)
        remainder = n_timepoints % 100
        assert remainder < 10 or remainder > 90, \
            f"Epoch length ({n_timepoints}) should be close to a round number (multiple of 100)"


class TestDataLoaderDeterminism:
    """Test determinism and reproducibility of data loading."""

    def test_multiple_loads_produce_same_data(self):
        """
        Test that loading data multiple times produces identical results.

        CRITICAL: Verifies data loading is deterministic (for reproducibility).
        """
        data1, labels1 = load_demo_preprocessed()
        data2, labels2 = load_demo_preprocessed()

        # Data should be identical
        assert np.allclose(data1, data2, atol=1e-10), \
            "Multiple loads should produce identical data"

        # Note: Labels may differ if single-event-code random split is used
        # But at minimum, shapes should match
        assert labels1.shape == labels2.shape, \
            "Multiple loads should produce same label shapes"

    def test_data_order_consistent_across_loads(self):
        """
        Test that epoch order is consistent across loads.

        CRITICAL: Verifies no unexpected shuffling occurs.
        """
        data1, labels1 = load_demo_preprocessed()
        data2, labels2 = load_demo_preprocessed()

        # First epoch should be identical
        assert np.allclose(data1[0], data2[0], atol=1e-10), \
            "First epoch should be identical across loads (no shuffling)"

        # Last epoch should be identical
        assert np.allclose(data1[-1], data2[-1], atol=1e-10), \
            "Last epoch should be identical across loads (no shuffling)"
