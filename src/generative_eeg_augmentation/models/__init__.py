"""
Models module for generative EEG augmentation.

Contains GAN and VAE architectures for synthetic EEG generation.
"""

from .gan import (
    EEGGenerator,
    EEGDiscriminator,
    EEGDiscriminatorEnhanced,
    load_generator,
    load_discriminator
)

from .vae import (
    VAE,
    load_vae
)

__all__ = [
    "EEGGenerator",
    "EEGDiscriminator",
    "EEGDiscriminatorEnhanced",
    "load_generator",
    "load_discriminator",
    "VAE",
    "load_vae",
]
