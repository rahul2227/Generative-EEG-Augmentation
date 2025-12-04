# Project Context for Claude Code

## Project Overview

This repository implements generative models for EEG (Electroencephalography) data augmentation and source localization. The project was developed for the course "Generative Neural Networks for the Sciences" and focuses on generating realistic, statistically valid synthetic EEG signals using advanced machine learning techniques.

### Primary Objectives
- Generate synthetic EEG signals using GANs and VAEs
- Validate generated signals through statistical and spectral analysis
- Enable data augmentation for research and practical applications
- Explore conditional generation based on EEG signal characteristics

## Technical Stack

### Core Dependencies
- **Python**: 3.10.15
- **Deep Learning**: PyTorch 2.5.1, TensorFlow 2.19.0, Keras 3.9.0
- **EEG Processing**: MNE 1.8.0
- **Scientific Computing**: NumPy 2.1.3, SciPy 1.15.1, Pandas 2.2.3
- **Visualization**: Matplotlib 3.10.0, Seaborn 0.13.2, Plotly 6.0.0
- **Development**: Jupyter Lab 4.3.4

### Environment Setup
The project uses conda for environment management. Environment specification: `environment.yml`

## Repository Structure

```
.
├── exploratory notebooks/       # Development and experimentation notebooks
│   ├── conditional_wasserstein_gan.ipynb
│   ├── Generative_Modelling_VAE.ipynb
│   ├── simple_gan_approach.ipynb
│   ├── data_preprocessing_module.ipynb
│   └── edge_case_analysis for preprocesing.ipynb
├── src/
│   ├── gan module/             # GAN implementation modules
│   └── preprocessor module/    # Data preprocessing utilities
│       ├── preprocessor_all_data.py
│       └── Preprocessing-legacy.py
├── validation module/          # Model validation and evaluation
│   ├── FID_EEG EValuation.ipynb
│   ├── Spectral and Temporal EValuation.ipynb
│   ├── VAE_FID_EVAL.ipynb
│   ├── VAE_SPECTRAL&TEMPORAL_EVAL.ipynb
│   └── feature_extractor.ipynb
├── environment.yml
└── README.md
```

## Key Components

### 1. Generative Models

#### Conditional Wasserstein GAN (CWGAN)
- Implementation: `exploratory notebooks/conditional_wasserstein_gan.ipynb`
- Uses Wasserstein loss with gradient penalty
- Conditional generation based on EEG characteristics
- Purpose: Generate realistic EEG waveforms with improved training stability

#### Variational Autoencoder (VAE)
- Implementation: `exploratory notebooks/Generative_Modelling_VAE.ipynb`
- Conditional VAE variant implemented
- Purpose: Learn latent representations and generate diverse EEG samples

#### Simple GAN
- Implementation: `exploratory notebooks/simple_gan_approach.ipynb`
- Baseline GAN implementation for comparison

### 2. Data Preprocessing
- Location: `src/preprocessor module/`
- Handles EEG signal cleaning, filtering, and normalization
- Edge case handling documented in dedicated notebook
- Two versions: current (`preprocessor_all_data.py`) and legacy

### 3. Validation Framework
- Location: `validation module/`
- **FID (Fréchet Inception Distance)**: Measures distribution similarity between real and generated EEG
- **Spectral Analysis**: Validates frequency domain characteristics
- **Temporal Analysis**: Validates time domain properties
- Separate evaluation pipelines for GAN and VAE outputs

## Working Conventions

### Code Style
- Jupyter notebooks for experimentation and visualization
- Python scripts for reusable modules
- Clear separation between exploration and production code

### Model Validation
- Multi-metric validation approach (FID, spectral, temporal)
- Both qualitative (visual) and quantitative evaluation
- Separate validation for each generative model type

### Data Flow
1. Raw EEG data → Preprocessing module
2. Preprocessed data → Generative model training
3. Generated samples → Validation module
4. Results → Analysis and visualization

## Current Branch
- Working in git worktree: `brave-wu`
- Main repository: `/Users/rahul/PycharmProjects/Generative-EEG-Augmentation`
- Main branch: `main`

## Development Notes

### Active Areas
- Conditional generation for both GANs and VAEs
- Comprehensive validation across multiple domains
- Preprocessing pipeline with edge case handling

### Key Considerations
- EEG signals have specific temporal and spectral characteristics that must be preserved
- Validation must ensure biological plausibility, not just statistical similarity
- Multiple models implemented for comparison and ensemble approaches

## Common Tasks

### Running Notebooks
Notebooks are organized by purpose:
- Start with `data_preprocessing_module.ipynb` for data preparation
- Use model-specific notebooks for training and generation
- Use validation module notebooks for evaluation

### Model Training
- Models are trained using Jupyter notebooks
- Training hyperparameters and architecture details in respective notebooks
- Both PyTorch and TensorFlow/Keras implementations available

### Validation
- Run FID evaluation notebooks after generation
- Run spectral/temporal analysis for domain-specific validation
- Compare results across different generative models

## Notes for Claude
- This is a research project with emphasis on validation and biological plausibility
- Multiple approaches (GAN, VAE, Wasserstein variants) implemented for comparison
- Notebooks contain both code and extensive analysis/visualization
- Preprocessing is critical - EEG data requires careful handling
- Validation metrics go beyond standard GAN evaluation (FID only) to include domain-specific measures
