"""ToneWave: generates a time-domain signal from (frequency, amplitude, harmonic_count) parameters."""

import torch
import torch.nn as nn
import math


class ToneWave(nn.Module):
    """Generates a time-domain signal from wave parameters.

    Each tone wave is defined by:
        - frequency (f): fundamental frequency
        - amplitude (A): signal strength
        - harmonic_softness (H_soft): continuous relaxation of harmonic count

    The signal is: A * sum_{k=1}^{K_max} sigmoid(H_soft - k) * (1/k) * sin(2*pi*k*f*t)

    The sigmoid envelope smoothly gates harmonics on/off, making H_soft differentiable.
    """

    def __init__(self, signal_length: int = 1024, k_max: int = 16):
        super().__init__()
        self.signal_length = signal_length
        self.k_max = k_max

        # Pre-compute time steps (not trainable)
        t = torch.linspace(0, 1, signal_length, dtype=torch.float32)
        self.register_buffer("t", t)

    def forward(self, f: torch.Tensor, A: torch.Tensor, H_soft: torch.Tensor) -> torch.Tensor:
        """Generate time-domain signals from wave parameters.

        Args:
            f: frequencies, shape (batch, num_waves)
            A: amplitudes, shape (batch, num_waves)
            H_soft: soft harmonic counts, shape (batch, num_waves)

        Returns:
            Signal tensor, shape (batch, num_waves, signal_length)
        """
        # t: (L,) -> (1, 1, L)
        t = self.t.view(1, 1, self.signal_length)

        # f: (batch, num_waves) -> (batch, num_waves, 1) for broadcasting with t
        f_exp = f.unsqueeze(-1)

        # Accumulate harmonics in a loop to avoid materializing (batch, waves, k_max, L) tensor
        signal = torch.zeros(f.shape[0], f.shape[1], self.signal_length,
                             device=f.device, dtype=f.dtype)

        for k in range(1, self.k_max + 1):
            # envelope: sigmoid(H_soft - k), shape (batch, num_waves)
            envelope_k = torch.sigmoid(H_soft - k).unsqueeze(-1)  # (batch, num_waves, 1)
            decay_k = 1.0 / k
            sinusoid_k = torch.sin(2 * math.pi * k * f_exp * t)  # (batch, num_waves, L)
            signal = signal + envelope_k * decay_k * sinusoid_k

        # Apply amplitude: (batch, num_waves, 1) * (batch, num_waves, L)
        signal = A.unsqueeze(-1) * signal

        return signal
