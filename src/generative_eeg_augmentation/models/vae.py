"""
VAE Models for EEG Generation

This module contains the Variational Autoencoder architecture for EEG generation.
Extracted from exploratory notebooks/Generative_Modelling_VAE.ipynb.
"""

import torch
import torch.nn as nn
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder for EEG data with convolutional encoder/decoder.
    
    This architecture uses 1D convolutions to process EEG signals, extracting
    spatial-temporal features into a latent representation and reconstructing
    the signal from the latent space.
    
    Args:
        n_channels: Number of EEG channels (default: 63)
        n_samples: Number of time samples per channel (default: 1001)
        latent_dim: Dimension of latent space (default: 16)
    
    Example:
        >>> vae = VAE(n_channels=63, n_samples=1001, latent_dim=16)
        >>> x = torch.randn(10, 63, 1001)
        >>> reconstructed, mu, logvar = vae(x)
        >>> reconstructed.shape
        torch.Size([10, 63, 1001])
    """
    
    def __init__(
        self,
        n_channels: int = 63,
        n_samples: int = 1001,
        latent_dim: int = 16
    ):
        super(VAE, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        
        # Encoder: Extracts latent features from input EEG data
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space representation (mu and logvar for reparameterization)
        self.fc_mu = nn.Linear(32 * n_samples, latent_dim)
        self.fc_logvar = nn.Linear(32 * n_samples, latent_dim)
        
        # Decoder: Reconstructs EEG data from latent representations
        self.decoder_fc = nn.Linear(latent_dim, 32 * n_samples)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=n_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the latent space.
        
        During training, samples from N(mu, std). During eval mode, returns mu directly
        for deterministic reconstruction.
        
        Args:
            mu: Mean of latent distribution, shape (batch_size, latent_dim)
            logvar: Log variance of latent distribution, shape (batch_size, latent_dim)
        
        Returns:
            Sampled latent vector z, shape (batch_size, latent_dim)
        
        Note:
            In eval mode, returns mu directly for deterministic behavior.
            In training mode, samples from N(mu, exp(0.5 * logvar)).
        """
        if not self.training:
            # In eval mode, use mean directly for deterministic reconstruction
            return mu
        
        # In training mode, use reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input EEG tensor of shape (batch_size, n_channels, n_samples)
        
        Returns:
            Tuple of (decoded, mu, logvar):
                - decoded: Reconstructed EEG, shape (batch_size, n_channels, n_samples)
                - mu: Latent mean, shape (batch_size, latent_dim)
                - logvar: Latent log variance, shape (batch_size, latent_dim)
        """
        # Encode to latent space
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode back to EEG space
        decoded = self.decoder_fc(z).view(-1, 32, self.n_samples)
        decoded = self.decoder(decoded)
        
        return decoded, mu, logvar


def load_vae(checkpoint_path: str, device: str = "cpu", input_dim: int = 63063, latent_dim: int = 128) -> VAE:
    """
    Load a pre-trained VAE from checkpoint.
    
    Note: This function is currently a placeholder as VAE checkpoints are not yet available.
    
    Args:
        checkpoint_path: Path to VAE checkpoint file
        device: PyTorch device string
        input_dim: Flattened dimension of input EEG (default: 63*1001=63063)
        latent_dim: Dimension of latent space (default: 128)
    
    Returns:
        Initialized VAE in eval mode with loaded weights
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist
        NotImplementedError: Currently not implemented pending checkpoint availability
    """
    raise NotImplementedError(
        "VAE checkpoint loading is not yet implemented. "
        "Please train a VAE model and save checkpoints first, or integrate "
        "the architecture from Generative_Modelling_VAE.ipynb notebook."
    )
