"""WaveEmbedding v7: Spectral State Language Model.

Core idea shift: tokens are not static waves — they are PERTURBATIONS
to a running spectral state. The spectral state evolves discretely at
each timestep as tokens modify it.

Spectral state = superposition of frequencies at 3 scales:
  - Long-range (low freq, slow decay): document topic
  - Mid-range (mid freq, mid decay): sentence/paragraph theme
  - Short-range (high freq, fast decay): local syntax

Each token learns a spectral perturbation:
  - delta_amp: (num_scales, K) — how it excites each frequency band
  - delta_phase: (num_scales, K) — how it rotates the phase

Training: masked language model (bidirectional).
  - Mask random tokens
  - Build forward + backward spectral states from unmasked tokens
  - At masked positions, score all vocab by how well their perturbation
    fits the combined spectral context
  - Cross-entropy loss

Params per token: num_scales * K * 2 (amp + phase perturbation)
  With 3 scales, 4 freqs each = 24 params/token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralState:
    """Running spectral state: amplitudes and phases at multiple frequency scales.

    State is a tuple of (amplitudes, phases), each (batch, num_scales, K).
    This is a plain data container, not an nn.Module.
    """

    def __init__(self, amplitudes: torch.Tensor, phases: torch.Tensor):
        self.amplitudes = amplitudes  # (batch, num_scales, K)
        self.phases = phases          # (batch, num_scales, K)

    @staticmethod
    def zeros(batch_size: int, num_scales: int, K: int,
              device: torch.device, dtype: torch.dtype = torch.float32):
        return SpectralState(
            amplitudes=torch.zeros(batch_size, num_scales, K, device=device, dtype=dtype),
            phases=torch.zeros(batch_size, num_scales, K, device=device, dtype=dtype),
        )

    def apply_perturbation(
        self,
        delta_amp: torch.Tensor,    # (batch, num_scales, K)
        delta_phase: torch.Tensor,  # (batch, num_scales, K)
        decay: torch.Tensor,        # (num_scales,)  — per-scale decay factor
    ) -> "SpectralState":
        """Apply a token's perturbation to the spectral state with per-scale decay.

        new_amp = decay * old_amp + delta_amp
        new_phase = decay * old_phase + delta_phase

        Decay is in (0, 1): long-range scales decay slowly, short-range fast.
        """
        decay_expanded = decay[None, :, None]  # (1, num_scales, 1)
        new_amp = decay_expanded * self.amplitudes + delta_amp
        new_phase = decay_expanded * self.phases + delta_phase
        return SpectralState(new_amp, new_phase)


class SpectralEmbedding(nn.Module):
    """Token perturbation embeddings for the spectral state model.

    Each token learns how it perturbs the spectral state:
      - delta_amp: (vocab, num_scales, K) — amplitude perturbation per scale/freq
      - delta_phase: (vocab, num_scales, K) — phase perturbation per scale/freq

    Params per token: num_scales * K * 2
    """

    def __init__(self, vocab_size: int, num_scales: int = 3, K: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_scales = num_scales
        self.K = K

        # Per-token perturbation parameters
        # Initialize amplitudes with small values, phases with small random
        self.delta_amp = nn.Parameter(
            torch.randn(vocab_size, num_scales, K) * 0.1
        )
        self.delta_phase = nn.Parameter(
            torch.randn(vocab_size, num_scales, K) * 0.3
        )

        # Per-scale decay rates (learned, constrained to (0,1) via sigmoid)
        # Initialize so long-range decays slowly, short-range decays fast
        # sigmoid(-2) ≈ 0.12 (fast decay), sigmoid(2) ≈ 0.88 (slow decay)
        init_decay_logits = torch.tensor([2.0, 0.0, -2.0])[:num_scales]
        self.decay_logits = nn.Parameter(init_decay_logits)

        # Per-scale base frequencies (fixed, not learned)
        # These define the frequency bands for each scale
        # Long-range: low freqs, short-range: high freqs
        base_freqs = []
        for s in range(num_scales):
            # Scale s gets frequencies in [2^s, 2^(s+1)] range
            freqs = torch.linspace(0.5 * (2 ** s), 2.0 * (2 ** s), K)
            base_freqs.append(freqs)
        self.register_buffer("base_freqs", torch.stack(base_freqs))  # (num_scales, K)

    @property
    def decay(self) -> torch.Tensor:
        """Per-scale decay factors in (0, 1)."""
        return torch.sigmoid(self.decay_logits)  # (num_scales,)

    def get_perturbation(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up the spectral perturbation for given token IDs.

        Args:
            ids: (...) token IDs, any shape

        Returns:
            delta_amp:   (..., num_scales, K)
            delta_phase: (..., num_scales, K)
        """
        return self.delta_amp[ids], self.delta_phase[ids]

    @property
    def params_per_token(self) -> int:
        return self.num_scales * self.K * 2


class SpectralStateLM(nn.Module):
    """Bidirectional Spectral State Language Model.

    Builds forward and backward spectral states from unmasked tokens,
    combines them, and scores masked positions against all vocab tokens.

    Scoring: spectral coherence — how well does a candidate token's
    perturbation align with the expected spectral state?

    Score(state, token) = sum over scales and freqs of:
        state_amp * token_delta_amp * cos(state_phase - token_delta_phase)

    This is analogous to wave interference energy but in the spectral
    state space rather than raw frequency space.
    """

    def __init__(self, vocab_size: int, num_scales: int = 3, K: int = 4):
        super().__init__()
        self.embedding = SpectralEmbedding(vocab_size, num_scales, K)

        # Learnable combination weights for forward/backward states
        self.combine_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

        # Temperature for logit scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _build_spectral_states(
        self,
        token_ids: torch.Tensor,     # (batch, seq_len)
        mask: torch.Tensor,           # (batch, seq_len) — True = masked
    ) -> tuple[list[SpectralState], list[SpectralState]]:
        """Build forward and backward spectral states from unmasked tokens.

        Returns lists of SpectralState, one per position.
        Forward[t] = state after processing positions 0..t-1 (excludes t).
        Backward[t] = state after processing positions t+1..T-1 (excludes t).
        """
        B, T = token_ids.shape
        S = self.embedding.num_scales
        K = self.embedding.K
        device = token_ids.device
        decay = self.embedding.decay  # (num_scales,)

        # Get all perturbations at once
        delta_amp, delta_phase = self.embedding.get_perturbation(token_ids)
        # (batch, seq_len, num_scales, K)

        # Forward pass: left to right
        fwd_states = []
        state = SpectralState.zeros(B, S, K, device)
        for t in range(T):
            fwd_states.append(state)  # state BEFORE position t
            if not mask[:, t].all():
                # Apply perturbation from unmasked tokens
                # For partially masked batches, zero out masked tokens' perturbations
                active = (~mask[:, t]).float()[:, None, None]  # (B, 1, 1)
                d_amp = delta_amp[:, t] * active
                d_phase = delta_phase[:, t] * active
                state = state.apply_perturbation(d_amp, d_phase, decay)

        # Backward pass: right to left
        bwd_states = [None] * T
        state = SpectralState.zeros(B, S, K, device)
        for t in range(T - 1, -1, -1):
            bwd_states[t] = state  # state BEFORE position t (from the right)
            if not mask[:, t].all():
                active = (~mask[:, t]).float()[:, None, None]
                d_amp = delta_amp[:, t] * active
                d_phase = delta_phase[:, t] * active
                state = state.apply_perturbation(d_amp, d_phase, decay)

        return fwd_states, bwd_states

    def _combine_states(
        self,
        fwd: SpectralState,
        bwd: SpectralState,
    ) -> SpectralState:
        """Combine forward and backward spectral states."""
        w = torch.sigmoid(self.combine_weight)
        combined_amp = w * fwd.amplitudes + (1 - w) * bwd.amplitudes
        combined_phase = w * fwd.phases + (1 - w) * bwd.phases
        return SpectralState(combined_amp, combined_phase)

    def _score_tokens(
        self,
        state: SpectralState,
    ) -> torch.Tensor:
        """Score all vocab tokens against the spectral state.

        Score = sum_s sum_k state_amp[s,k] * delta_amp[v,s,k] * cos(state_phase[s,k] - delta_phase[v,s,k])

        This measures spectral coherence: how much the candidate token's
        perturbation is "in phase" with what the context expects.

        Args:
            state: SpectralState with amplitudes/phases (batch, num_scales, K)

        Returns:
            logits: (batch, vocab_size)
        """
        # state.amplitudes: (B, S, K)
        # self.embedding.delta_amp: (V, S, K)
        # state.phases: (B, S, K)
        # self.embedding.delta_phase: (V, S, K)

        s_amp = state.amplitudes      # (B, S, K)
        s_phase = state.phases        # (B, S, K)
        v_amp = self.embedding.delta_amp    # (V, S, K)
        v_phase = self.embedding.delta_phase  # (V, S, K)

        # Broadcast: (B, 1, S, K) * (1, V, S, K) * cos(...)  → (B, V, S, K) → sum → (B, V)
        phase_diff = s_phase.unsqueeze(1) - v_phase.unsqueeze(0)  # (B, V, S, K)
        coherence = (
            s_amp.unsqueeze(1) * v_amp.unsqueeze(0) * torch.cos(phase_diff)
        )  # (B, V, S, K)

        logits = coherence.sum(dim=(-2, -1))  # (B, V)
        return logits / (self.temperature.abs() + 1e-6)

    def forward(
        self,
        token_ids: torch.Tensor,   # (batch, seq_len)
        mask: torch.Tensor,         # (batch, seq_len) — True = masked positions
    ) -> torch.Tensor:
        """Compute logits at masked positions.

        Args:
            token_ids: (batch, seq_len) — includes true tokens at masked positions
                       (needed for loss, not used for state building)
            mask: (batch, seq_len) — True where tokens are masked

        Returns:
            logits: (num_masked, vocab_size) — predictions at masked positions
        """
        B, T = token_ids.shape

        # Build bidirectional spectral states
        fwd_states, bwd_states = self._build_spectral_states(token_ids, mask)

        # Score masked positions
        all_logits = []
        for t in range(T):
            if mask[:, t].any():
                # Combine forward and backward states at this position
                combined = self._combine_states(fwd_states[t], bwd_states[t])

                # Score all vocab tokens
                logits = self._score_tokens(combined)  # (B, V)

                # Only keep logits for actually-masked samples in this batch
                masked_in_batch = mask[:, t]  # (B,)
                all_logits.append(logits[masked_in_batch])

        if all_logits:
            return torch.cat(all_logits, dim=0)  # (num_masked, V)
        else:
            return torch.zeros(0, self.embedding.vocab_size,
                               device=token_ids.device)

    def get_targets(
        self,
        token_ids: torch.Tensor,  # (batch, seq_len)
        mask: torch.Tensor,       # (batch, seq_len)
    ) -> torch.Tensor:
        """Extract true token IDs at masked positions (for cross-entropy loss).

        Returns:
            targets: (num_masked,)
        """
        return token_ids[mask]  # flattened masked token IDs

    def spectral_similarity(
        self,
        id1: torch.Tensor,
        id2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between two tokens via their perturbation alignment.

        For evaluation / comparison with v5/v6.
        """
        d_amp1, d_phase1 = self.embedding.get_perturbation(id1)
        d_amp2, d_phase2 = self.embedding.get_perturbation(id2)

        phase_diff = d_phase1 - d_phase2
        coherence = (d_amp1 * d_amp2 * torch.cos(phase_diff)).sum(dim=(-2, -1))

        # Normalize
        norm1 = (d_amp1 ** 2).sum(dim=(-2, -1)).sqrt()
        norm2 = (d_amp2 ** 2).sum(dim=(-2, -1)).sqrt()
        return coherence / (norm1 * norm2 + 1e-8)


def mask_tokens(
    token_ids: torch.Tensor,
    mask_prob: float = 0.15,
    pad_id: int = 0,
    unk_id: int = 1,
) -> torch.Tensor:
    """Create MLM mask. Don't mask special tokens (pad, unk).

    Args:
        token_ids: (batch, seq_len)
        mask_prob: probability of masking each token

    Returns:
        mask: (batch, seq_len) — True = masked
    """
    mask = torch.rand_like(token_ids.float()) < mask_prob
    # Don't mask special tokens
    mask = mask & (token_ids != pad_id) & (token_ids != unk_id)
    return mask
