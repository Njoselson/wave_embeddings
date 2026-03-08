"""WaveEmbedding v5: One frequency, one amplitude per token. Harmonics for depth.

Each token is one note — a single frequency with a single amplitude.
Harmonics (overtones) create richness: z(t) = Σ_h (A/h^d) * exp(j·2π·h·f·t)

Two tokens resonate when their harmonics align. A token at f=100 and a token
at f=200 interact through harmonic overlap (2×100 = 200). This creates
multi-scale relationships from minimal parameters.

A sequence's "feeling" is the running interference of all token waves.
Early tokens keep resonating — waves don't forget. Long-range dependency
comes from harmonic resonance, not explicit attention.

Inference: look up 2 numbers per token. Compute similarity via sinc.
The trained model is a table of (frequency, amplitude) pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveEmbeddingV5(nn.Module):
    """Each token = 1 frequency + 1 amplitude. Harmonics expand these into
    a rich signal with H overtones, but the embedding is still just 2 numbers.

    Parameters per token: 2
    Effective frequencies per token: H (from harmonics)
    Global parameters: 1 (harmonic decay)
    """

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_harmonics = num_harmonics

        # 2 params per token. This IS the embedding.
        self.frequencies = nn.Parameter(torch.randn(vocab_size) * 3.0)
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size) + torch.randn(vocab_size) * 0.1
        )

        # Global harmonic decay — controls overtone richness
        # Higher decay = purer tone, lower = richer harmonics
        self.decay = nn.Parameter(torch.tensor(1.5))

    def get_harmonics(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand token params into harmonic series.

        Args:
            ids: (...) token IDs, any shape

        Returns:
            freqs: (..., H) harmonic frequencies [f, 2f, 3f, ..., Hf]
            amps:  (..., H) harmonic amplitudes  [A, A/2^d, A/3^d, ..., A/H^d]
        """
        f = self.frequencies[ids]  # (...)
        A = self.amplitudes[ids]   # (...)

        h = torch.arange(
            1, self.num_harmonics + 1, device=f.device, dtype=f.dtype
        )  # (H,)

        # Broadcast: (...) × (H,) → (..., H)
        freqs = f.unsqueeze(-1) * h
        amps = A.unsqueeze(-1) / (h ** self.decay)

        return freqs, amps


# ---- Analytical energy/similarity (same sinc formula as v4) ----

def energy(
    f1: torch.Tensor, A1: torch.Tensor,
    f2: torch.Tensor, A2: torch.Tensor,
) -> torch.Tensor:
    """Analytical interference energy between two harmonic wave sets.

    E = ∫₀¹ |z₁(t) + z₂(t)|² dt  (exact via sinc)

    Args:
        f1, A1: (batch, H) harmonics for set 1
        f2, A2: (batch, H) harmonics for set 2

    Returns:
        (batch,) energy
    """
    f_all = torch.cat([f1, f2], dim=-1)
    A_all = torch.cat([A1, A2], dim=-1)
    df = f_all.unsqueeze(-1) - f_all.unsqueeze(-2)
    amp_outer = A_all.unsqueeze(-1) * A_all.unsqueeze(-2)
    return (amp_outer * torch.sinc(2.0 * df)).sum(dim=(-2, -1))


def self_energy(f: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Self-energy of a single harmonic wave set."""
    df = f.unsqueeze(-1) - f.unsqueeze(-2)
    amp_outer = A.unsqueeze(-1) * A.unsqueeze(-2)
    return (amp_outer * torch.sinc(2.0 * df)).sum(dim=(-2, -1))


def similarity(
    f1: torch.Tensor, A1: torch.Tensor,
    f2: torch.Tensor, A2: torch.Tensor,
) -> torch.Tensor:
    """Normalized cross-term similarity. ~1 for identical, ~0 for unrelated."""
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
    return -F.logsigmoid(pos_energy).mean() + -F.logsigmoid(-neg_energy).mean()


# ---- Models ----

class SkipGramV5(nn.Module):
    """Word-level skip-gram with harmonic wave interference.

    Takes token IDs directly (not characters).
    Similarity = analytical harmonic interference energy.
    """

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.embedding = WaveEmbeddingV5(vocab_size, num_harmonics)

    def forward(
        self,
        target_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_ids:   (batch,) target token IDs
            positive_ids: (batch,) positive context IDs
            negative_ids: (batch, num_neg) negative IDs

        Returns:
            pos_energy: (batch,)
            neg_energy: (batch, num_neg)
        """
        t_f, t_A = self.embedding.get_harmonics(target_ids)
        p_f, p_A = self.embedding.get_harmonics(positive_ids)
        pos_energy = energy(t_f, t_A, p_f, p_A)

        batch, num_neg = negative_ids.shape
        H = self.embedding.num_harmonics

        t_f_exp = t_f.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, H)
        t_A_exp = t_A.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, H)
        n_f, n_A = self.embedding.get_harmonics(negative_ids.reshape(-1))

        neg_energy = energy(t_f_exp, t_A_exp, n_f, n_A).reshape(batch, num_neg)

        return pos_energy, neg_energy

    def running_wave_score(
        self,
        history_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates by resonance with accumulated history.

        The running wave = superposition of all history token harmonics.
        Score = cross-term interference between running wave and candidate.
        This is the "feeling" — how well the candidate fits what came before.

        Args:
            history_ids:   (seq_len,) tokens seen so far
            candidate_ids: (num_candidates,) tokens to score

        Returns:
            (num_candidates,) resonance scores
        """
        # Running wave = all harmonics from all history tokens
        h_f, h_A = self.embedding.get_harmonics(history_ids)  # (seq_len, H)
        running_f = h_f.reshape(-1)  # (seq_len * H,)
        running_A = h_A.reshape(-1)

        # Score each candidate
        c_f, c_A = self.embedding.get_harmonics(candidate_ids)  # (num_cand, H)

        # Cross-term: how much each candidate resonates with running wave
        # For each candidate i: Σ_r Σ_h A_r * A_{c,h} * sinc(2*(f_r - f_{c,h}))
        # = sum over all (running, candidate) harmonic pairs
        # Vectorized: (num_cand, seq*H, H) → expensive but exact
        df = running_f.unsqueeze(0).unsqueeze(-1) - c_f.unsqueeze(1)  # (C, R, H)
        amp_cross = running_A.unsqueeze(0).unsqueeze(-1) * c_A.unsqueeze(1)  # (C, R, H)
        scores = 2.0 * (amp_cross * torch.sinc(2.0 * df)).sum(dim=(-2, -1))  # (C,)

        return scores


class WaveLM(nn.Module):
    """Language model where context = running wave, prediction = resonance.

    At each position t, the running wave is the superposition of all
    previous token harmonics. Next-token logits = cross-term interference
    between running wave and every vocab token's harmonics.

    No matrices. No attention. Just wave accumulation and sinc.

    Forward cost: O(T × V × H²) sinc calls — comparable to one
    transformer attention layer for typical sizes.
    """

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.embedding = WaveEmbeddingV5(vocab_size, num_harmonics)

    def _cross_terms_chunked(self, seq_f, seq_A, all_f, all_A, chunk_size):
        """Compute per-position cross-terms with all vocab, in chunks."""
        B, T, H = seq_f.shape
        V = all_f.shape[0]
        C = torch.zeros(B, T, V, device=seq_f.device, dtype=seq_f.dtype)

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            sf = seq_f[:, t_start:t_end]
            sA = seq_A[:, t_start:t_end]
            df = sf[:, :, :, None, None] - all_f[None, None, None, :, :]
            amp = sA[:, :, :, None, None] * all_A[None, None, None, :, :]
            C[:, t_start:t_end, :] = 2.0 * (amp * torch.sinc(2.0 * df)).sum(dim=(2, 4))

        return C

    def forward(
        self,
        input_ids: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Compute next-token logits via running wave interference.

        Args:
            input_ids: (batch, seq_len) token IDs
            chunk_size: positions to process at once (memory/speed tradeoff).
                        Larger = faster on GPU but more VRAM.

        Returns:
            logits: (batch, seq_len, vocab_size) — logits[t] predicts token at t
                    using history from positions 0..t-1
        """
        B, T = input_ids.shape
        V = self.embedding.vocab_size
        H = self.embedding.num_harmonics

        # All vocab harmonics (computed once)
        all_ids = torch.arange(V, device=input_ids.device)
        all_f, all_A = self.embedding.get_harmonics(all_ids)  # (V, H)

        # Sequence token harmonics
        seq_f, seq_A = self.embedding.get_harmonics(input_ids)  # (B, T, H)

        # Cross-term of each position with all vocab tokens
        C = self._cross_terms_chunked(seq_f, seq_A, all_f, all_A, chunk_size)

        # Causal cumsum: logits[t] = sum of C[0..t-1]
        cum_C = C.cumsum(dim=1)
        logits = torch.zeros_like(cum_C)
        logits[:, 1:, :] = cum_C[:, :-1, :]

        return logits

