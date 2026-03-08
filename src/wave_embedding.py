"""WaveTokenEmbedding: maps token IDs to wave parameters and generates spectral embeddings."""

import torch
import torch.nn as nn
import math

from .tone_wave import ToneWave


class WaveTokenEmbedding(nn.Module):
    """Maps each token in a vocabulary to a set of tone wave parameters.

    Each token gets `num_waves` tone waves, each with 3 learnable scalars:
        - frequency (f): stored as normalized values [0, 1], scaled to Hz internally
        - amplitude (A)
        - harmonic count (H_soft, continuous relaxation)

    The embedding pipeline:
        token_id -> (f, A, H) params -> time-domain signal -> FFT -> spectral embedding
    """

    def __init__(
        self,
        vocab_size: int,
        num_waves: int = 7,
        signal_length: int = 1024,
        k_max: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves
        self.signal_length = signal_length
        self.k_max = k_max
        # real + imag for each FFT bin
        self.embed_dim = (signal_length // 2 + 1) * 2

        self.tone_wave = ToneWave(signal_length=signal_length, k_max=k_max)

        # Frequencies: free parameters initialized log-spaced in [1, max_freq].
        # Log-spacing gives equal relative resolution across the spectrum.
        max_freq = signal_length / 4
        self.frequencies = nn.Parameter(
            torch.exp(torch.linspace(0, math.log(max_freq), vocab_size * num_waves))
            .view(vocab_size, num_waves)
            + torch.randn(vocab_size, num_waves) * 0.5
        )

        # Amplitudes: initialize near 1.0
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size, num_waves) + torch.randn(vocab_size, num_waves) * 0.1
        )

        # Harmonic counts (soft): initialize around 4-5 harmonics
        self.harmonics = nn.Parameter(
            torch.full((vocab_size, num_waves), 4.5) + torch.randn(vocab_size, num_waves) * 0.5
        )

    def get_wave_params(self, token_ids: torch.Tensor):
        """Look up wave parameters for given token IDs.

        Returns:
            f, A, H_soft each of shape (*token_ids.shape, num_waves)
            f is in Hz (softplus ensures positive)
        """
        # softplus keeps frequencies positive with well-behaved gradients everywhere
        f = torch.nn.functional.softplus(self.frequencies[token_ids])
        A = self.amplitudes[token_ids]
        H = self.harmonics[token_ids]
        return f, A, H

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate spectral embeddings for tokens.

        Args:
            token_ids: (batch, seq_len) tensor of token indices

        Returns:
            Spectral embeddings of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = token_ids.shape

        # Flatten to (batch * seq_len,) for lookup
        flat_ids = token_ids.view(-1)
        f, A, H = self.get_wave_params(flat_ids)

        # Generate time-domain signals: (batch * seq_len, num_waves, signal_length)
        signals = self.tone_wave(f, A, H)

        # Sum across waves to get composite signal per token: (batch * seq_len, signal_length)
        composite = signals.sum(dim=1)

        # FFT to get spectral representation: (batch * seq_len, fft_bins) complex
        spectrum = torch.fft.rfft(composite, dim=-1)

        # Use real + imag (preserves phase info, better gradient flow than abs)
        embedding = torch.cat([spectrum.real, spectrum.imag], dim=-1)

        # Reshape back: (batch, seq_len, embed_dim)
        embedding = embedding.view(batch_size, seq_len, self.embed_dim)

        return embedding
