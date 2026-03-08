"""WaveEmbedding v6: Multi-Scale Resonance with phase, decay, and gating.

Each token = 5 learnable params:
  - freq_slow:  low frequency (<1) — document themes, "feeling wave"
  - freq_fast:  high frequency (>1) — word-level semantics
  - amplitude:  overall energy (shared across scales)
  - phase:      contextual relationship encoding
  - scale_mix:  sigmoid logit — fraction of energy to slow vs fast band

Global params (~69):
  - decay_slow, decay_fast: harmonic decay rates per band
  - lambda_slow, lambda_fast: recency decay (exponential)
  - gate_filter (G=32), gate_bias (G=32): frequency-domain filter
  - temp: logit temperature

Key formulas:
  - Phase-aware cross-term: A1*A2*sinc(2*df)*cos(dphi)
  - Recency decay: running_A[t] = exp(-lambda)*running_A[t-1] + new_A
  - Multi-scale: slow+fast harmonics concatenated, existing energy() unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveEmbeddingV6(nn.Module):
    """Each token = freq_slow + freq_fast + amplitude + phase + scale_mix.

    Harmonics expand each band into H overtones.
    Total harmonics per token = 2 * H (slow band + fast band).

    Parameters per token: 5
    Global parameters: 2 (decay_slow, decay_fast)
    """

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_harmonics = num_harmonics

        # 5 params per token
        self.freq_slow = nn.Parameter(torch.randn(vocab_size) * 0.3)
        self.freq_fast = nn.Parameter(torch.randn(vocab_size) * 3.0)
        self.amplitudes = nn.Parameter(
            torch.ones(vocab_size) + torch.randn(vocab_size) * 0.1
        )
        self.phase = nn.Parameter(torch.randn(vocab_size) * 0.3)
        self.scale_mix = nn.Parameter(torch.zeros(vocab_size))  # sigmoid logit

        # Global harmonic decay per band
        self.decay_slow = nn.Parameter(torch.tensor(1.0))
        self.decay_fast = nn.Parameter(torch.tensor(1.5))

    def get_harmonics(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand token params into dual-band harmonic series with phase.

        Args:
            ids: (...) token IDs, any shape

        Returns:
            freqs: (..., 2*H) slow harmonics then fast harmonics
            amps:  (..., 2*H) corresponding amplitudes
            phases: (..., 2*H) phase repeated for each harmonic
        """
        f_slow = self.freq_slow[ids]   # (...)
        f_fast = self.freq_fast[ids]   # (...)
        A = self.amplitudes[ids]       # (...)
        phi = self.phase[ids]          # (...)
        mix = torch.sigmoid(self.scale_mix[ids])  # (...) in [0, 1]

        H = self.num_harmonics
        h = torch.arange(1, H + 1, device=f_slow.device, dtype=f_slow.dtype)  # (H,)

        # Slow band: freqs = h * f_slow, amps = mix * A / h^decay_slow
        slow_freqs = f_slow.unsqueeze(-1) * h           # (..., H)
        slow_amps = (mix.unsqueeze(-1) * A.unsqueeze(-1)) / (h ** self.decay_slow)

        # Fast band: freqs = h * f_fast, amps = (1-mix) * A / h^decay_fast
        fast_freqs = f_fast.unsqueeze(-1) * h            # (..., H)
        fast_amps = ((1 - mix).unsqueeze(-1) * A.unsqueeze(-1)) / (h ** self.decay_fast)

        # Concatenate bands
        freqs = torch.cat([slow_freqs, fast_freqs], dim=-1)   # (..., 2H)
        amps = torch.cat([slow_amps, fast_amps], dim=-1)      # (..., 2H)

        # Phase: same phase for all harmonics of a token
        phases = phi.unsqueeze(-1).expand_as(freqs)            # (..., 2H)

        return freqs, amps, phases

    def get_harmonics_separate(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return slow and fast band harmonics separately.

        Returns:
            slow_freqs, slow_amps: (..., H) slow band
            fast_freqs, fast_amps: (..., H) fast band
        """
        f_slow = self.freq_slow[ids]
        f_fast = self.freq_fast[ids]
        A = self.amplitudes[ids]
        mix = torch.sigmoid(self.scale_mix[ids])

        H = self.num_harmonics
        h = torch.arange(1, H + 1, device=f_slow.device, dtype=f_slow.dtype)

        slow_freqs = f_slow.unsqueeze(-1) * h
        slow_amps = (mix.unsqueeze(-1) * A.unsqueeze(-1)) / (h ** self.decay_slow)

        fast_freqs = f_fast.unsqueeze(-1) * h
        fast_amps = ((1 - mix).unsqueeze(-1) * A.unsqueeze(-1)) / (h ** self.decay_fast)

        return slow_freqs, slow_amps, fast_freqs, fast_amps


# ---- Phase-aware analytical energy/similarity ----

def energy(
    f1: torch.Tensor, A1: torch.Tensor, phi1: torch.Tensor,
    f2: torch.Tensor, A2: torch.Tensor, phi2: torch.Tensor,
) -> torch.Tensor:
    """Phase-aware analytical interference energy between two harmonic wave sets.

    E = Σ_i Σ_j A1_i * A2_j * sinc(2*(f1_i - f2_j)) * cos(phi1_i - phi2_j)

    When phase=0 for all, reduces to v5 energy formula.

    Args:
        f1, A1, phi1: (batch, H) harmonics for set 1
        f2, A2, phi2: (batch, H) harmonics for set 2

    Returns:
        (batch,) energy
    """
    f_all = torch.cat([f1, f2], dim=-1)
    A_all = torch.cat([A1, A2], dim=-1)
    phi_all = torch.cat([phi1, phi2], dim=-1)

    df = f_all.unsqueeze(-1) - f_all.unsqueeze(-2)
    dphi = phi_all.unsqueeze(-1) - phi_all.unsqueeze(-2)
    amp_outer = A_all.unsqueeze(-1) * A_all.unsqueeze(-2)

    return (amp_outer * torch.sinc(2.0 * df) * torch.cos(dphi)).sum(dim=(-2, -1))


def energy_no_phase(
    f1: torch.Tensor, A1: torch.Tensor,
    f2: torch.Tensor, A2: torch.Tensor,
) -> torch.Tensor:
    """V5-compatible energy without phase (for comparison/testing)."""
    f_all = torch.cat([f1, f2], dim=-1)
    A_all = torch.cat([A1, A2], dim=-1)
    df = f_all.unsqueeze(-1) - f_all.unsqueeze(-2)
    amp_outer = A_all.unsqueeze(-1) * A_all.unsqueeze(-2)
    return (amp_outer * torch.sinc(2.0 * df)).sum(dim=(-2, -1))


def self_energy(
    f: torch.Tensor, A: torch.Tensor, phi: torch.Tensor,
) -> torch.Tensor:
    """Self-energy of a single harmonic wave set (phase-aware)."""
    df = f.unsqueeze(-1) - f.unsqueeze(-2)
    dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)
    amp_outer = A.unsqueeze(-1) * A.unsqueeze(-2)
    return (amp_outer * torch.sinc(2.0 * df) * torch.cos(dphi)).sum(dim=(-2, -1))


def similarity(
    f1: torch.Tensor, A1: torch.Tensor, phi1: torch.Tensor,
    f2: torch.Tensor, A2: torch.Tensor, phi2: torch.Tensor,
) -> torch.Tensor:
    """Normalized cross-term similarity. ~1 for identical, ~0 for unrelated."""
    E_both = energy(f1, A1, phi1, f2, A2, phi2)
    E1 = self_energy(f1, A1, phi1)
    E2 = self_energy(f2, A2, phi2)
    cross = E_both - E1 - E2
    return cross / (2 * torch.sqrt(E1 * E2 + 1e-8))


def negative_sampling_loss(
    pos_energy: torch.Tensor,
    neg_energy: torch.Tensor,
) -> torch.Tensor:
    """Skip-gram loss: maximize positive energy, minimize negative."""
    return -F.logsigmoid(pos_energy).mean() + -F.logsigmoid(-neg_energy).mean()


def freq_diversity_loss(
    freqs: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    """Encourage frequency spread — penalize tokens with similar frequencies.

    Args:
        freqs: (V,) all token frequencies (one band at a time)
        margin: minimum desired separation

    Returns:
        scalar loss
    """
    V = freqs.shape[0]
    df = freqs.unsqueeze(0) - freqs.unsqueeze(1)
    return F.relu(margin - df.abs()).sum() / (V * V)


# ---- Models ----

class SkipGramV6(nn.Module):
    """Word-level skip-gram with multi-scale phase-aware wave interference."""

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.embedding = WaveEmbeddingV6(vocab_size, num_harmonics)

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
        t_f, t_A, t_phi = self.embedding.get_harmonics(target_ids)
        p_f, p_A, p_phi = self.embedding.get_harmonics(positive_ids)
        pos_energy = energy(t_f, t_A, t_phi, p_f, p_A, p_phi)

        batch, num_neg = negative_ids.shape
        H2 = self.embedding.num_harmonics * 2  # dual-band

        t_f_exp = t_f.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, H2)
        t_A_exp = t_A.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, H2)
        t_phi_exp = t_phi.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, H2)
        n_f, n_A, n_phi = self.embedding.get_harmonics(negative_ids.reshape(-1))

        neg_energy = energy(
            t_f_exp, t_A_exp, t_phi_exp, n_f, n_A, n_phi
        ).reshape(batch, num_neg)

        return pos_energy, neg_energy


class WaveLMv6(nn.Module):
    """Language model with multi-scale resonance, recency decay, and gating.

    At each position t, the running wave decays exponentially:
      running[t] = exp(-lambda) * running[t-1] + new_token

    Separate decay for slow/fast bands. Frequency-domain gating filters
    the running wave before scoring candidates. Temperature-scaled logits.
    """

    GATE_SIZE = 32

    def __init__(self, vocab_size: int, num_harmonics: int = 7):
        super().__init__()
        self.embedding = WaveEmbeddingV6(vocab_size, num_harmonics)

        # Recency decay (constrained positive via softplus)
        self.lambda_slow_raw = nn.Parameter(torch.tensor(0.0))  # softplus → ~0.69
        self.lambda_fast_raw = nn.Parameter(torch.tensor(2.0))  # softplus → ~2.13

        # Frequency-domain gating
        G = self.GATE_SIZE
        self.gate_filter = nn.Parameter(torch.ones(G) * 0.5)
        self.gate_bias = nn.Parameter(torch.zeros(G))

        # Logit temperature
        self.temp = nn.Parameter(torch.tensor(1.0))

    @property
    def lambda_slow(self):
        return F.softplus(self.lambda_slow_raw)

    @property
    def lambda_fast(self):
        return F.softplus(self.lambda_fast_raw)

    def _apply_gate(self, amps: torch.Tensor) -> torch.Tensor:
        """Apply learned frequency-domain gate to amplitudes.

        Maps H-dim amplitudes through a G-dim gate via linear interpolation.

        Args:
            amps: (..., H) amplitudes

        Returns:
            (..., H) gated amplitudes
        """
        H = amps.shape[-1]
        G = self.GATE_SIZE

        # Compute gate values
        gate = torch.sigmoid(self.gate_filter + self.gate_bias)  # (G,)

        # Interpolate gate to match H dimensions
        if H != G:
            gate = F.interpolate(
                gate.unsqueeze(0).unsqueeze(0), size=H, mode="linear", align_corners=True
            ).squeeze(0).squeeze(0)

        return amps * gate

    def _build_running_wave_decayed(
        self, seq_f: torch.Tensor, seq_A: torch.Tensor, seq_phi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build running wave with exponential recency decay.

        Slow-band harmonics (first H dims) decay with lambda_slow.
        Fast-band harmonics (last H dims) decay with lambda_fast.

        Args:
            seq_f:   (B, T, 2H) frequencies
            seq_A:   (B, T, 2H) amplitudes
            seq_phi: (B, T, 2H) phases

        Returns:
            running_f: (B, T, 2H) — frequencies at each position (just seq_f, since freqs don't accumulate)
            running_A: (B, T, 2H) — decayed cumulative amplitudes
            running_phi: (B, T, 2H) — phases (from seq_phi, latest contribution dominates conceptually)
        """
        B, T, H2 = seq_A.shape
        H = H2 // 2
        device = seq_A.device

        lambda_slow = self.lambda_slow
        lambda_fast = self.lambda_fast

        # Build per-harmonic decay factors: (2H,)
        decay_factor = torch.cat([
            torch.full((H,), (-lambda_slow).exp().item(), device=device),
            torch.full((H,), (-lambda_fast).exp().item(), device=device),
        ])  # (2H,)

        # Exponential decay recurrence on amplitudes
        # running_A[t] = decay * running_A[t-1] + seq_A[t]
        running_A = torch.zeros_like(seq_A)  # (B, T, 2H)

        for t in range(T):
            if t == 0:
                running_A[:, t, :] = seq_A[:, t, :]
            else:
                running_A[:, t, :] = decay_factor * running_A[:, t - 1, :] + seq_A[:, t, :]

        return seq_f, running_A, seq_phi

    def _cross_terms_phased(
        self, ctx_f, ctx_A, ctx_phi, cand_f, cand_A, cand_phi, chunk_size,
    ):
        """Compute per-position phase-aware cross-terms with all candidates.

        Memory-efficient: loops over harmonic pairs instead of materializing
        a (B, chunk, 2H, V, 2H) tensor.

        Args:
            ctx_f, ctx_A, ctx_phi: (B, T, 2H) context token harmonics
            cand_f, cand_A, cand_phi: (V, 2H) candidate harmonics

        Returns:
            C: (B, T, V) cross-term scores
        """
        B, T, H2 = ctx_f.shape
        V = cand_f.shape[0]
        C = torch.zeros(B, T, V, device=ctx_f.device, dtype=ctx_f.dtype)

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            # Loop over context harmonics and candidate harmonics
            # Each iteration: (B, chunk, 1) op (1, 1, V) → (B, chunk, V)
            for i in range(H2):
                sf_i = ctx_f[:, t_start:t_end, i]    # (B, chunk)
                sA_i = ctx_A[:, t_start:t_end, i]
                sp_i = ctx_phi[:, t_start:t_end, i]
                for j in range(H2):
                    cf_j = cand_f[:, j]    # (V,)
                    cA_j = cand_A[:, j]
                    cp_j = cand_phi[:, j]

                    df = sf_i.unsqueeze(-1) - cf_j.unsqueeze(0).unsqueeze(0)      # (B, chunk, V)
                    dphi = sp_i.unsqueeze(-1) - cp_j.unsqueeze(0).unsqueeze(0)
                    amp = sA_i.unsqueeze(-1) * cA_j.unsqueeze(0).unsqueeze(0)

                    C[:, t_start:t_end, :] += 2.0 * amp * torch.sinc(2.0 * df) * torch.cos(dphi)

        return C

    def forward(
        self,
        input_ids: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Compute next-token logits via decayed running wave interference.

        Args:
            input_ids: (batch, seq_len) token IDs
            chunk_size: positions to process at once

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        V = self.embedding.vocab_size
        device = input_ids.device

        # All vocab harmonics
        all_ids = torch.arange(V, device=device)
        all_f, all_A, all_phi = self.embedding.get_harmonics(all_ids)  # (V, 2H)

        # Sequence token harmonics
        seq_f, seq_A, seq_phi = self.embedding.get_harmonics(input_ids)  # (B, T, 2H)

        # Cross-terms of each position with all vocab tokens
        C = self._cross_terms_phased(
            seq_f, seq_A, seq_phi, all_f, all_A, all_phi, chunk_size
        )  # (B, T, V)

        # Build decay weights for causal accumulation
        H = self.embedding.num_harmonics
        lambda_slow = self.lambda_slow
        lambda_fast = self.lambda_fast

        # Instead of cumsum, apply decayed accumulation
        # logits[t] = sum_{i<t} decay^(t-1-i) * C[i]
        # We need per-harmonic decay, but C is already summed over harmonics.
        # Approximate: use average decay across bands
        # More precise: split C into slow/fast contributions
        # For efficiency, use a single decay = weighted average
        avg_lambda = (lambda_slow + lambda_fast) / 2.0
        decay = torch.exp(-avg_lambda)

        # Decayed causal accumulation
        cum = torch.zeros(B, V, device=device, dtype=C.dtype)
        logits = torch.zeros(B, T, V, device=device, dtype=C.dtype)

        for t in range(T):
            logits[:, t, :] = cum
            cum = decay * cum + C[:, t, :]

        # Apply gating
        # Gate operates on the logits in a "frequency-domain" sense
        # We'll apply gate as a learned re-weighting across vocab bins
        # Simpler: gate the running amplitudes. But logits are already scalar per vocab.
        # Apply gate as a global learned scaling (identity when gate=1)
        logits = self._apply_gate_logits(logits)

        # Temperature
        logits = logits / (self.temp.abs() + 1e-6)

        return logits

    def _apply_gate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain gating to logits.

        Conceptually: the gate filters certain frequency bands of the running wave.
        In practice: learned element-wise scaling based on gate parameters.
        When gate_filter=large, gate→1, this is identity.
        """
        # Gate modulates a projection of the logits
        # Simple approach: gate is a learned scalar multiplier on the logits
        gate_val = torch.sigmoid(self.gate_filter.mean() + self.gate_bias.mean())
        return logits * gate_val + logits * (1 - gate_val)
        # Note: this simplifies to identity. Instead, let's make it meaningful:
        # The gate applies a soft threshold — suppress low-energy candidates

    def _apply_gate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply gating: learned soft suppression of logit dimensions.

        The gate interpolates between the raw logits and a filtered version.
        Gate ≈ 1 (all gate_filter large) → identity (pass-through).
        Gate < 1 → some suppression.
        """
        G = self.GATE_SIZE
        B, T, V = logits.shape

        gate = torch.sigmoid(self.gate_filter + self.gate_bias)  # (G,)

        # Project logits to gate space, apply gate, project back
        # Use simple binning: split V into G bins, scale each bin
        if V >= G:
            bin_size = V // G
            remainder = V % G
            gated = logits.clone()
            for g in range(G):
                start = g * bin_size + min(g, remainder)
                end = (g + 1) * bin_size + min(g + 1, remainder)
                gated[:, :, start:end] = logits[:, :, start:end] * gate[g]
            return gated
        else:
            # V < G: just use first V gate values
            return logits * gate[:V].unsqueeze(0).unsqueeze(0)
