"""WaveEmbedding v2: frequency-first architecture.

Key changes from v1:
- Each token = small number of waves (default 3), each defined by (f, A)
- Harmonics fixed at 1 (just fundamental) or very few
- Embedding IS the wave parameters, not FFT bins
- Interference computed directly between token wave params
- Much lower dimensional: 3 waves * 2 params = 6 scalars per token
"""

import torch
import torch.nn as nn
import math


class WaveEmbeddingV2(nn.Module):
    """Frequency-first wave embeddings.

    Each token is represented by `num_waves` waves, each with:
        - frequency (f): the primary carrier of meaning
        - amplitude (A): salience/strength

    Sequence composition computes pairwise wave interference directly,
    forcing frequencies to learn meaningful structure.
    """

    def __init__(
        self,
        vocab_size: int,
        num_waves: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves
        # Output: num_waves frequencies + num_waves amplitudes
        self.embed_dim = num_waves * 2

        # Frequencies: log-spaced init, free to learn
        self.frequencies = nn.Parameter(
            torch.randn(vocab_size, num_waves) * 2.0
        )

        # Amplitudes: init near 1
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size, num_waves) + torch.randn(vocab_size, num_waves) * 0.1
        )

    def get_params(self, token_ids: torch.Tensor):
        """Returns (f, A) for given token IDs.

        f: softplus-transformed frequencies (always positive)
        A: raw amplitudes
        """
        f = torch.nn.functional.softplus(self.frequencies[token_ids])
        A = self.amplitudes[token_ids]
        return f, A

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get wave parameter embeddings.

        Args:
            token_ids: (batch, seq_len)

        Returns:
            (batch, seq_len, num_waves * 2) — concatenated [f, A]
        """
        f, A = self.get_params(token_ids)
        return torch.cat([f, A], dim=-1)

    def interference(self, token_ids: torch.Tensor, sample_points: int = 64) -> torch.Tensor:
        """Compute interference pattern for a sequence.

        Instead of FFT, directly compute what happens when all token waves
        are superimposed and sampled at a few points. This creates a
        low-dimensional "interference fingerprint" of the sequence.

        Args:
            token_ids: (batch, seq_len)
            sample_points: number of points to sample the composed wave

        Returns:
            (batch, sample_points) — the interference pattern
        """
        batch_size, seq_len = token_ids.shape
        f, A = self.get_params(token_ids)
        # f, A: (batch, seq_len, num_waves)

        # Sample points in [0, 1]
        t = torch.linspace(0, 1, sample_points, device=f.device, dtype=f.dtype)
        # t: (P,) -> (1, 1, 1, P)
        t = t.view(1, 1, 1, sample_points)

        # f: (batch, seq_len, num_waves) -> (batch, seq_len, num_waves, 1)
        f_exp = f.unsqueeze(-1)
        A_exp = A.unsqueeze(-1)

        # Each wave contributes: A * sin(2*pi*f*t)
        # shape: (batch, seq_len, num_waves, P)
        waves = A_exp * torch.sin(2 * math.pi * f_exp * t)

        # Sum over waves and tokens -> interference pattern
        # (batch, seq_len, num_waves, P) -> (batch, P)
        pattern = waves.sum(dim=(1, 2))

        return pattern


class WaveModelV2(nn.Module):
    """Wave model v2: frequency-first, minimal parameters."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_waves: int = 3,
        sample_points: int = 64,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.wave_embedding = WaveEmbeddingV2(
            vocab_size=vocab_size,
            num_waves=num_waves,
        )
        self.sample_points = sample_points

        # Tiny classifier — can't compensate for lazy frequencies
        self.classifier = nn.Sequential(
            nn.Linear(sample_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward: tokens -> wave interference pattern -> classify."""
        pattern = self.wave_embedding.interference(token_ids, self.sample_points)
        return self.classifier(pattern)
