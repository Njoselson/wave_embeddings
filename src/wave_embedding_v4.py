"""WaveEmbedding v4: Simplified. The embedding IS the parameters.

Core idea: each token is K waves, each wave = (frequency, amplitude).
That's 2K numbers per token. Similarity is computed analytically via sinc —
no sampling, no FFT, no materialization needed.

Inference = table lookup + a handful of sinc calls. No GPU required.
A trained model is literally a CSV of floats.

For word composition from characters: stack character wave params with
a position-dependent frequency shift so "cat" ≠ "act".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveEmbeddingV4(nn.Module):
    """Parameter table of wave embeddings. That's it.

    Each token → K waves → 2K learnable scalars (frequency, amplitude).
    All similarity/energy computations are analytical — no time-domain needed.
    """

    def __init__(self, vocab_size: int, num_waves: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_waves = num_waves

        self.frequencies = nn.Parameter(torch.randn(vocab_size, num_waves) * 3.0)
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size, num_waves) + torch.randn(vocab_size, num_waves) * 0.1
        )

        # Position frequency: how much character position shifts wave frequency
        # Enables "cat" ≠ "act" when composing words from characters
        self.position_freq = nn.Parameter(torch.tensor(0.1))

    def get_params(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up wave parameters. This IS the embedding."""
        return self.frequencies[ids], self.amplitudes[ids]

    def get_word_params(
        self,
        char_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose word params from characters by stacking with position shifts.

        A word's waves = all character waves, with each character's frequencies
        shifted by position * beta. Result is (batch, num_chars * num_waves) for
        both f and A — still just a flat set of (f, A) pairs.

        Args:
            char_ids: (batch, max_len) character IDs
            mask: (batch, max_len) boolean, True = valid char

        Returns:
            f_word: (batch, max_len * num_waves) flattened frequencies
            A_word: (batch, max_len * num_waves) flattened amplitudes
        """
        f = self.frequencies[char_ids]  # (batch, L, K)
        A = self.amplitudes[char_ids]   # (batch, L, K)

        # Position shift: each char position shifts frequency
        batch, L, K = f.shape
        positions = torch.arange(L, device=f.device, dtype=f.dtype)
        pos_shift = (positions * self.position_freq).view(1, L, 1)  # (1, L, 1)
        f = f + pos_shift

        # Zero out padding
        if mask is not None:
            A = A * mask.unsqueeze(-1).float()

        # Flatten chars × waves → single set of waves
        return f.reshape(batch, -1), A.reshape(batch, -1)


def energy(
    f1: torch.Tensor,
    A1: torch.Tensor,
    f2: torch.Tensor,
    A2: torch.Tensor,
) -> torch.Tensor:
    """Analytical interference energy between two sets of waves.

    E = integral_0^1 |z1(t) + z2(t)|^2 dt

    Computed exactly via sinc cross-terms. No sampling.
    For K waves each, this is (2K)^2 sinc evaluations.

    Args:
        f1, A1: (batch, K1) wave params for set 1
        f2, A2: (batch, K2) wave params for set 2

    Returns:
        (batch,) energy per pair
    """
    # Stack all waves from both sets
    f_all = torch.cat([f1, f2], dim=-1)  # (batch, K1+K2)
    A_all = torch.cat([A1, A2], dim=-1)

    # E = sum_i sum_j A_i * A_j * sinc(2 * (f_i - f_j))
    # When i==j: sinc(0) = 1, giving A_i^2
    # When i!=j: cross-term interference
    df = f_all.unsqueeze(-1) - f_all.unsqueeze(-2)  # (batch, N, N)
    amp_outer = A_all.unsqueeze(-1) * A_all.unsqueeze(-2)  # (batch, N, N)
    return (amp_outer * torch.sinc(2.0 * df)).sum(dim=(-2, -1))


def self_energy(f: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Self-energy of a single wave set."""
    df = f.unsqueeze(-1) - f.unsqueeze(-2)
    amp_outer = A.unsqueeze(-1) * A.unsqueeze(-2)
    return (amp_outer * torch.sinc(2.0 * df)).sum(dim=(-2, -1))


def similarity(
    f1: torch.Tensor,
    A1: torch.Tensor,
    f2: torch.Tensor,
    A2: torch.Tensor,
) -> torch.Tensor:
    """Normalized similarity: cross-term / (2 * sqrt(E1 * E2)).

    Returns ~1 for identical wave sets, ~0 for unrelated.
    """
    E_both = energy(f1, A1, f2, A2)
    E1 = self_energy(f1, A1)
    E2 = self_energy(f2, A2)
    cross = E_both - E1 - E2
    return cross / (2 * torch.sqrt(E1 * E2 + 1e-8))


def negative_sampling_loss(
    pos_energy: torch.Tensor,
    neg_energy: torch.Tensor,
) -> torch.Tensor:
    """Skip-gram loss: maximize positive energy, minimize negative."""
    pos_loss = -F.logsigmoid(pos_energy).mean()
    neg_loss = -F.logsigmoid(-neg_energy).mean()
    return pos_loss + neg_loss


class SkipGramV4(nn.Module):
    """Word-level skip-gram using character wave composition.

    Characters → wave params → stack with position shift → analytical energy.
    Entire forward pass is just table lookups + sinc. No neural network layers.
    """

    def __init__(self, char_vocab_size: int, num_waves: int = 3):
        super().__init__()
        self.embedding = WaveEmbeddingV4(
            vocab_size=char_vocab_size,
            num_waves=num_waves,
        )

    def forward(
        self,
        target_chars: torch.Tensor,
        pos_chars: torch.Tensor,
        neg_chars: torch.Tensor,
        target_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute analytical energy for positive and negative word pairs."""
        t_f, t_A = self.embedding.get_word_params(target_chars, target_mask)
        p_f, p_A = self.embedding.get_word_params(pos_chars, pos_mask)

        pos_energy = energy(t_f, t_A, p_f, p_A)

        # Negatives: (batch, num_neg, max_len)
        batch_size, num_neg, max_len_n = neg_chars.shape
        max_len_t = target_chars.shape[1]

        # Expand target for each negative
        target_exp = target_chars.unsqueeze(1).expand(-1, num_neg, -1)
        tmask_exp = target_mask.unsqueeze(1).expand(-1, num_neg, -1)

        # Flatten to (batch*num_neg, ...)
        te_f, te_A = self.embedding.get_word_params(
            target_exp.reshape(-1, max_len_t),
            tmask_exp.reshape(-1, max_len_t),
        )
        n_f, n_A = self.embedding.get_word_params(
            neg_chars.reshape(-1, max_len_n),
            neg_mask.reshape(-1, max_len_n),
        )

        neg_energy = energy(te_f, te_A, n_f, n_A).reshape(batch_size, num_neg)

        return pos_energy, neg_energy

    def materialize(self, ids: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """Optional: produce a dense vector for downstream classifiers.

        Evaluates waves at `dim` evenly-spaced time points, summing over
        all tokens (interference). Only needed if feeding into a traditional NN.

        Args:
            ids: (batch, seq_len) token IDs
            dim: number of sample points

        Returns:
            (batch, dim*2) — real and imag of interference pattern
        """
        import math

        f, A = self.embedding.get_params(ids)  # (batch, seq_len, K)
        t = torch.linspace(0, 1, dim, device=f.device, dtype=f.dtype)

        # phase: (batch, seq_len, K, dim)
        phase = 2 * math.pi * f.unsqueeze(-1) * t.view(1, 1, 1, dim)
        # Sum over seq_len and K → (batch, dim)
        real = (A.unsqueeze(-1) * torch.cos(phase)).sum(dim=(1, 2))
        imag = (A.unsqueeze(-1) * torch.sin(phase)).sum(dim=(1, 2))
        return torch.cat([real, imag], dim=-1)  # (batch, dim*2)
