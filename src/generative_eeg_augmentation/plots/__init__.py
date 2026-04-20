"""
Visualization functions for EEG data analysis.

This module provides plotting functions for comparing real and synthetic EEG data
across multiple domains (time, frequency, statistics). All functions return
matplotlib Figure objects suitable for notebook display or Streamlit embedding.
"""

from generative_eeg_augmentation.plots.eeg_visualizations import (
    plot_statistical_comparison,
    plot_waveforms,
    plot_psd_comparison,
    plot_band_power_comparison,
    plot_similarity_heatmap,
    set_eeg_visualization_style,
    EEG_BANDS
)

__all__ = [
    'plot_statistical_comparison',
    'plot_waveforms',
    'plot_psd_comparison',
    'plot_band_power_comparison',
    'plot_similarity_heatmap',
    'set_eeg_visualization_style',
    'EEG_BANDS'
]
