"""WaveEmbedding v3: Complex exponential surrogate for learnable frequencies.

Key insight from Hayes et al. (2022): sin(wt) has a non-convex loss landscape
w.r.t. frequency with dense local minima. exp(j*w*t) provides a smooth loss
landscape where frequency can actually be learned via gradient descent.

The complex exponential gives the optimizer a path through the complex plane,
avoiding the oscillatory traps of real-valued sinusoids.
"""

import torch
import torch.nn as nn
import math


class WaveEmbeddingV3(nn.Module):
    """Complex-exponential wave embeddings with learnable frequencies.

    Each token is represented by `num_waves` complex exponentials:
        z_k(t) = A_k * exp(j * 2π * f_k * t)

    The interference pattern is computed by summing all token waves
    and sampling at discrete points. The real and imaginary parts
    of the resulting signal form the embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        num_waves: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves

        # Frequencies: init with diverse spread
        # Use randn * scale to get good initial diversity
        self.frequencies = nn.Parameter(
            torch.randn(vocab_size, num_waves) * 3.0
        )

        # Amplitudes: init near 1
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size, num_waves) + torch.randn(vocab_size, num_waves) * 0.1
        )

    @property
    def embed_dim(self):
        return self.num_waves * 2  # real + imag per wave

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get wave parameter embeddings (f, A concatenated)."""
        f = self.frequencies[token_ids]
        A = self.amplitudes[token_ids]
        return torch.cat([f, A], dim=-1)

    def interference(self, token_ids: torch.Tensor, sample_points: int = 64) -> torch.Tensor:
        """Compute complex interference pattern for a sequence.

        Uses complex exponentials instead of sin() for smooth frequency gradients.

        Args:
            token_ids: (batch, seq_len)
            sample_points: number of points to sample

        Returns:
            (batch, sample_points * 2) — real and imag of interference pattern
        """
        batch_size, seq_len = token_ids.shape

        f = self.frequencies[token_ids]  # (batch, seq_len, num_waves)
        A = self.amplitudes[token_ids]   # (batch, seq_len, num_waves)

        # Sample points in [0, 1]
        t = torch.linspace(0, 1, sample_points, device=f.device, dtype=f.dtype)
        t = t.view(1, 1, 1, sample_points)  # (1, 1, 1, P)

        f_exp = f.unsqueeze(-1)  # (batch, seq_len, num_waves, 1)
        A_exp = A.unsqueeze(-1)  # (batch, seq_len, num_waves, 1)

        # Complex exponential: A * exp(j * 2π * f * t)
        # phase: (batch, seq_len, num_waves, P)
        phase = 2 * math.pi * f_exp * t

        # Complex signal components
        real = A_exp * torch.cos(phase)
        imag = A_exp * torch.sin(phase)

        # Sum over waves and tokens -> interference
        # (batch, seq_len, num_waves, P) -> (batch, P)
        real_sum = real.sum(dim=(1, 2))
        imag_sum = imag.sum(dim=(1, 2))

        # Concatenate real and imag as features
        pattern = torch.cat([real_sum, imag_sum], dim=-1)  # (batch, P*2)

        return pattern


class HarmonicWaveEmbedding(nn.Module):
    """Wave embeddings with learnable harmonics and position encoding.

    Each token has K waves, each with:
      - frequency f_k (fundamental)
      - amplitude A_k
      - harmonic decay d_k (controls overtone rolloff: 1/h^d)

    A token's signal includes harmonics:
      z_k(t) = A_k * Σ_{h=1}^{H} (1/h^d_k) * exp(j·2π·h·f_k·t)

    For word composition, characters are summed with position-dependent
    phase shifts so that character order matters:
      z_word(t) = Σ_i z_{c_i}(t) * exp(j·2π·i·β)
    """

    def __init__(
        self,
        vocab_size: int,
        num_waves: int = 3,
        num_harmonics: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves
        self.num_harmonics = num_harmonics

        # Frequencies: init with diverse spread
        self.frequencies = nn.Parameter(
            torch.randn(vocab_size, num_waves) * 3.0
        )

        # Amplitudes: init near 1
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size, num_waves) + torch.randn(vocab_size, num_waves) * 0.1
        )

        # Harmonic decay: init near 1.5 (moderate rolloff)
        # Higher d = faster harmonic decay = purer tone
        # Lower d = richer harmonics
        self.decays = nn.Parameter(
            torch.ones(vocab_size, num_waves) * 1.5 + torch.randn(vocab_size, num_waves) * 0.1
        )

        # Position frequency: scalar controlling how much position shifts phase
        self.position_freq = nn.Parameter(torch.tensor(0.1))

    @property
    def embed_dim(self):
        return self.num_waves * 3  # f, A, d per wave


class WaveModelV3(nn.Module):
    """Wave model v3: complex exponential, frequency-first."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_waves: int = 3,
        sample_points: int = 64,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.wave_embedding = WaveEmbeddingV3(
            vocab_size=vocab_size,
            num_waves=num_waves,
        )
        self.sample_points = sample_points

        # Tiny classifier
        self.classifier = nn.Sequential(
            nn.Linear(sample_points * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pattern = self.wave_embedding.interference(token_ids, self.sample_points)
        return self.classifier(pattern)
