"""
GAN Models for EEG Generation

This module contains the generator and discriminator architectures for
conditional Wasserstein GAN with gradient penalty (CWGAN-GP).
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn


class EEGGenerator(nn.Module):
    """
    Conditional EEG Generator using transposed convolutions.
    
    Generates synthetic EEG epochs from random noise conditioned on class labels.
    The architecture upsamples from latent space through transposed 1D convolutions
    to produce multi-channel EEG signals of specified length.
    
    Args:
        latent_dim: Dimension of the input noise vector (default: 100)
        n_channels: Number of EEG channels to generate (default: 63)
        target_signal_len: Length of generated signal in samples (default: 1001)
        num_classes: Number of conditional classes (default: 2)
    
    Example:
        >>> gen = EEGGenerator()
        >>> noise = torch.randn(10, 100)
        >>> labels = torch.zeros(10, 2)
        >>> labels[:, 0] = 1  # Set first class
        >>> synthetic_eeg = gen(noise, labels)
        >>> synthetic_eeg.shape
        torch.Size([10, 63, 1001])
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        n_channels: int = 63,
        target_signal_len: int = 1001,
        num_classes: int = 2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.target_signal_len = target_signal_len
        self.num_classes = num_classes

        # Calculate base length for upsampling (3 ConvTranspose1d layers with stride=2 → 8x upsampling)
        self.base_length = target_signal_len // 8
        remainder = target_signal_len - (self.base_length * 8)

        # Fully connected layer to project latent+label to feature map
        self.fc = nn.Linear(latent_dim + num_classes, 256 * self.base_length)

        # Transposed convolution blocks for upsampling
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose1d(64, n_channels, kernel_size=4, stride=2, padding=1, output_padding=remainder),
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic EEG from noise and labels.
        
        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
        
        Returns:
            Generated EEG tensor of shape (batch_size, n_channels, target_signal_len)
        """
        # Concatenate noise and labels
        x = torch.cat([noise, labels], dim=1)
        
        # Project to feature map
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.base_length)
        
        # Upsample through transposed convolutions
        out = self.deconv_blocks(x)
        return out


class EEGDiscriminator(nn.Module):
    """
    Conditional EEG Discriminator (Critic) for Wasserstein GAN.
    
    Evaluates the quality and authenticity of EEG signals conditioned on class labels.
    Uses 1D convolutions to downsample the signal and produce a scalar score.
    
    Args:
        n_channels: Number of EEG channels (default: 63)
        target_signal_len: Length of input signal in samples (default: 1001)
        num_classes: Number of conditional classes (default: 2)
    
    Example:
        >>> disc = EEGDiscriminator()
        >>> eeg = torch.randn(10, 63, 1001)
        >>> labels = torch.zeros(10, 2)
        >>> labels[:, 1] = 1  # Set second class
        >>> score = disc(eeg, labels)
        >>> score.shape
        torch.Size([10, 1])
    """
    
    def __init__(
        self,
        n_channels: int = 63,
        target_signal_len: int = 1001,
        num_classes: int = 2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.target_signal_len = target_signal_len
        self.num_classes = num_classes

        # Convolutional blocks for downsampling
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_channels + num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv1d(128, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator score for EEG data.
        
        Args:
            eeg: EEG tensor of shape (batch_size, n_channels, signal_len)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
        
        Returns:
            Discriminator scores of shape (batch_size, 1)
        """
        # Expand labels to match signal length and concatenate with EEG
        label_expand = labels.unsqueeze(2).repeat(1, 1, eeg.shape[2])
        d_in = torch.cat([eeg, label_expand], dim=1)
        
        # Downsample and compute score
        out = self.conv_blocks(d_in)
        return out.mean(dim=2)


class EEGDiscriminatorEnhanced(nn.Module):
    """
    Enhanced Conditional EEG Discriminator with additional layers.
    
    An improved discriminator architecture with more capacity to better distinguish
    real and synthetic EEG signals, particularly for addressing amplitude and
    frequency realism issues.
    
    Args:
        n_channels: Number of EEG channels (default: 63)
        target_signal_len: Length of input signal in samples (default: 1001)
        num_classes: Number of conditional classes (default: 2)
    
    Example:
        >>> disc = EEGDiscriminatorEnhanced()
        >>> eeg = torch.randn(10, 63, 1001)
        >>> labels = torch.zeros(10, 2)
        >>> score = disc(eeg, labels)
        >>> score.shape
        torch.Size([10, 1])
    """
    
    def __init__(
        self,
        n_channels: int = 63,
        target_signal_len: int = 1001,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Enhanced convolutional blocks with additional layer
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_channels + num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, eeg: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator score for EEG data.
        
        Args:
            eeg: EEG tensor of shape (batch_size, n_channels, signal_len)
            labels: One-hot encoded labels of shape (batch_size, num_classes)
        
        Returns:
            Discriminator scores of shape (batch_size, 1)
        """
        # Expand labels to match signal length and concatenate with EEG
        labels_expand = labels.unsqueeze(2).repeat(1, 1, eeg.shape[2])
        d_in = torch.cat([eeg, labels_expand], dim=1)
        
        # Downsample and compute score
        out = self.conv_blocks(d_in)
        return out.mean(dim=2)


def load_generator(
    model_variant: str = "original",
    device: str = "cpu",
    latent_dim: int = 100,
    n_channels: int = 63,
    target_signal_len: int = 1001,
    num_classes: int = 2
) -> EEGGenerator:
    """
    Load a pre-trained EEG generator from saved checkpoint.
    
    This function creates an EEGGenerator instance and loads weights from
    the appropriate checkpoint file based on the model variant.
    
    Args:
        model_variant: One of "original" or "enhanced". Determines checkpoint path.
        device: PyTorch device string ("cpu", "cuda", "mps").
        latent_dim: Dimension of latent noise vector (default: 100).
        n_channels: Number of EEG channels (default: 63).
        target_signal_len: Length of generated signal in samples (default: 1001).
        num_classes: Number of conditional classes (default: 2).
    
    Returns:
        Initialized EEGGenerator in eval mode with loaded weights.
    
    Raises:
        ValueError: If model_variant is not recognized.
        FileNotFoundError: If checkpoint file does not exist.
    
    Example:
        >>> gen = load_generator("original", device="cpu")
        >>> noise = torch.randn(10, 100)
        >>> labels = torch.zeros(10, 2)
        >>> labels[:, 0] = 1
        >>> synthetic_eeg = gen(noise, labels)
        >>> synthetic_eeg.shape
        torch.Size([10, 63, 1001])
    """
    # Determine checkpoint path based on variant
    if model_variant == "original":
        checkpoint_path = Path("exploratory notebooks/models/best_generator.pth")
    elif model_variant == "enhanced":
        checkpoint_path = Path("exploratory notebooks/models/enhanced/best_generator.pth")
    else:
        raise ValueError(
            f"Unknown model_variant: {model_variant}. "
            f"Must be one of: 'original', 'enhanced'"
        )
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure model checkpoints are in the correct location."
        )
    
    # Create model instance
    generator = EEGGenerator(
        latent_dim=latent_dim,
        n_channels=n_channels,
        target_signal_len=target_signal_len,
        num_classes=num_classes
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()
    
    return generator


def load_discriminator(
    model_variant: str = "original",
    enhanced: bool = False,
    device: str = "cpu",
    n_channels: int = 63,
    target_signal_len: int = 1001,
    num_classes: int = 2
) -> Union[EEGDiscriminator, EEGDiscriminatorEnhanced]:
    """
    Load a pre-trained EEG discriminator from saved checkpoint.
    
    Args:
        model_variant: One of "original" or "enhanced". Determines checkpoint path.
        enhanced: If True, load EEGDiscriminatorEnhanced architecture (default: False).
        device: PyTorch device string ("cpu", "cuda", "mps").
        n_channels: Number of EEG channels (default: 63).
        target_signal_len: Length of input signal in samples (default: 1001).
        num_classes: Number of conditional classes (default: 2).
    
    Returns:
        Initialized discriminator in eval mode with loaded weights.
    
    Raises:
        ValueError: If model_variant is not recognized.
        FileNotFoundError: If checkpoint file does not exist.
    
    Example:
        >>> disc = load_discriminator("original", device="cpu")
        >>> eeg = torch.randn(10, 63, 1001)
        >>> labels = torch.zeros(10, 2)
        >>> score = disc(eeg, labels)
    """
    # Determine checkpoint path based on variant
    if model_variant == "original":
        checkpoint_path = Path("exploratory notebooks/models/best_discriminator.pth")
    elif model_variant == "enhanced":
        checkpoint_path = Path("exploratory notebooks/models/enhanced/best_discriminator.pth")
    else:
        raise ValueError(
            f"Unknown model_variant: {model_variant}. "
            f"Must be one of: 'original', 'enhanced'"
        )
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure model checkpoints are in the correct location."
        )
    
    # Create model instance based on architecture type
    if enhanced or model_variant == "enhanced":
        discriminator = EEGDiscriminatorEnhanced(
            n_channels=n_channels,
            target_signal_len=target_signal_len,
            num_classes=num_classes
        )
    else:
        discriminator = EEGDiscriminator(
            n_channels=n_channels,
            target_signal_len=target_signal_len,
            num_classes=num_classes
        )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    discriminator.load_state_dict(checkpoint)
    discriminator.to(device)
    discriminator.eval()
    
    return discriminator
