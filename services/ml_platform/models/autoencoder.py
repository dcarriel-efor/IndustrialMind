"""
PyTorch Autoencoder for Sensor Anomaly Detection

Architecture:
- Encoder: Compress 4 sensor inputs to latent representation
- Decoder: Reconstruct original sensor values
- Anomaly Score: Reconstruction error (MSE)
"""

import torch
import torch.nn as nn
from typing import Tuple


class SensorAutoencoder(nn.Module):
    """
    Standard Autoencoder for multivariate sensor anomaly detection.

    Uses reconstruction error as anomaly score. Trained on normal patterns,
    anomalies will have higher reconstruction errors.

    Architecture:
        Input (4 sensors) → [64] → [32] → [latent_dim] → [32] → [64] → Output (4 sensors)
    """

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2
    ):
        """
        Initialize autoencoder architecture.

        Args:
            input_dim: Number of input features (4 sensors: temp, vib, pressure, power)
            latent_dim: Size of bottleneck/latent representation
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(SensorAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Bottleneck layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed, latent):
                - reconstructed: Decoded output (batch_size, input_dim)
                - latent: Encoded representation (batch_size, latent_dim)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error (MSE per sample).

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Reconstruction error per sample (batch_size,)
        """
        reconstructed, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

    def predict_anomaly(
        self,
        x: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict if samples are anomalies based on threshold.

        Args:
            x: Input tensor (batch_size, input_dim)
            threshold: Reconstruction error threshold

        Returns:
            Tuple of (is_anomaly, error):
                - is_anomaly: Boolean tensor (batch_size,)
                - error: Reconstruction error (batch_size,)
        """
        error = self.reconstruction_error(x)
        is_anomaly = error > threshold
        return is_anomaly, error


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for anomaly detection with uncertainty.

    VAE learns a probabilistic latent representation, useful for:
    - Uncertainty quantification
    - Better generalization to unseen normal patterns
    - Sampling new "normal" examples
    """

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout: float = 0.2
    ):
        """Initialize VAE architecture."""
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)  # Log variance

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Returns:
            Tuple of (mu, logvar):
                - mu: Mean of latent distribution (batch_size, latent_dim)
                - logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        Allows backpropagation through sampling operation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error per sample."""
        reconstructed, _, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


def vae_loss(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> torch.Tensor:
    """
    VAE loss = Reconstruction loss + beta * KL divergence.

    Args:
        x: Original input
        reconstructed: Reconstructed output
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL term (beta-VAE)

    Returns:
        Total loss (scalar)
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='mean')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / x.size(0)  # Average over batch

    return recon_loss + beta * kl_div


if __name__ == "__main__":
    # Quick test of model architectures
    print("Testing SensorAutoencoder...")
    model = SensorAutoencoder(input_dim=4, latent_dim=4)
    x = torch.randn(32, 4)  # Batch of 32 samples
    reconstructed, latent = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    error = model.reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean error: {error.mean():.4f}")

    print("\nTesting VariationalAutoencoder...")
    vae = VariationalAutoencoder(input_dim=4, latent_dim=4)
    reconstructed, mu, logvar = vae(x)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    loss = vae_loss(x, reconstructed, mu, logvar)
    print(f"VAE loss: {loss.item():.4f}")

    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
