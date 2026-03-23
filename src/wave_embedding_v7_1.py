"""WaveEmbedding v7.1: Gated Spectral State Language Model.

Builds on v7's spectral perturbation idea but adds content-dependent gating.

Key change from v7: the state update is no longer a blind linear recurrence.
Each token computes a GATE that controls how much of the existing state to
retain vs overwrite, based on the interaction between its perturbation and
the current state.

v7:   state = decay * state + delta[token]           # linear, fixed
v7.1: state = gate * state + (1 - gate) * delta      # nonlinear, adaptive

The gate is computed from the spectral coherence between the token's
perturbation and the current state — if the token is "in phase" with the
state (expected/reinforcing), the gate stays high (keep the state).
If the token is "out of phase" (surprising/new topic), the gate drops
(overwrite the state).

This gives the model selective memory: it can choose to remember "king"
while letting "of" and "the" pass through without much impact.

State evolution is still a recurrence (sequential), but the gating adds
the nonlinearity needed to break the expressiveness bottleneck.

Params per token: num_scales * K * 2 (same as v7 — no extra params)
The gate is computed from existing parameters, not learned separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralState:
    """Running spectral state: amplitudes and phases at multiple frequency scales."""

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

    def gated_update(
        self,
        delta_amp: torch.Tensor,    # (batch, num_scales, K)
        delta_phase: torch.Tensor,  # (batch, num_scales, K)
        gate_amp: torch.Tensor,     # (batch, num_scales, K) in (0, 1)
        gate_phase: torch.Tensor,   # (batch, num_scales, K) in (0, 1)
    ) -> "SpectralState":
        """Content-dependent gated update.

        gate ≈ 1: keep existing state (token reinforces context)
        gate ≈ 0: overwrite with token's perturbation (token shifts context)

        Separate gates for amplitude and phase — a token can shift the
        "what" (amplitude) without changing the "when" (phase), or vice versa.
        """
        new_amp = gate_amp * self.amplitudes + (1 - gate_amp) * delta_amp
        new_phase = gate_phase * self.phases + (1 - gate_phase) * delta_phase
        return SpectralState(new_amp, new_phase)


class GatedSpectralEmbedding(nn.Module):
    """Token embeddings with gating for selective state updates.

    Each token learns:
      - delta_amp: (vocab, num_scales, K) — amplitude perturbation
      - delta_phase: (vocab, num_scales, K) — phase perturbation
      - gate_bias: (vocab, num_scales) — per-token tendency to retain vs overwrite
        (high bias = "function word" that passes through, low = "content word" that
        overwrites the state)

    The actual gate is computed as:
      gate = sigmoid(gate_bias + coherence(state, delta))

    where coherence measures how well the token's perturbation aligns with
    the current state. This means:
      - A content word arriving in a new context (low coherence) → low gate → overwrite
      - A content word reinforcing existing context (high coherence) → high gate → reinforce
      - A function word (high gate_bias) → high gate regardless → pass through
    """

    def __init__(self, vocab_size: int, num_scales: int = 3, K: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_scales = num_scales
        self.K = K

        # Per-token perturbation parameters (same as v7)
        self.delta_amp = nn.Parameter(
            torch.randn(vocab_size, num_scales, K) * 0.1
        )
        self.delta_phase = nn.Parameter(
            torch.randn(vocab_size, num_scales, K) * 0.3
        )

        # Per-token gate bias: how much this token tends to preserve the state
        # Initialize near zero (sigmoid(0) = 0.5 = balanced)
        # Function words will learn positive bias (preserve), content words negative (overwrite)
        self.gate_bias = nn.Parameter(
            torch.zeros(vocab_size, num_scales)
        )

        # Learnable gate sensitivity: how much coherence affects the gate
        # Higher = more content-dependent gating
        self.gate_sensitivity = nn.Parameter(torch.tensor(1.0))

        # Per-scale decay (same as v7, used as a soft baseline)
        init_decay_logits = torch.tensor([2.0, 0.0, -2.0])[:num_scales]
        self.decay_logits = nn.Parameter(init_decay_logits)

        # Base frequencies (fixed, same as v7)
        base_freqs = []
        for s in range(num_scales):
            freqs = torch.linspace(0.5 * (2 ** s), 2.0 * (2 ** s), K)
            base_freqs.append(freqs)
        self.register_buffer("base_freqs", torch.stack(base_freqs))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logits)

    def get_perturbation(self, ids):
        return self.delta_amp[ids], self.delta_phase[ids]

    def get_gate_bias(self, ids):
        return self.gate_bias[ids]  # (..., num_scales)

    def compute_gate(
        self,
        state: SpectralState,
        delta_amp: torch.Tensor,    # (batch, num_scales, K)
        delta_phase: torch.Tensor,  # (batch, num_scales, K)
        gate_bias: torch.Tensor,    # (batch, num_scales)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute content-dependent gate from state-perturbation coherence.

        Coherence per scale = mean_k(state_amp * delta_amp * cos(state_phase - delta_phase))
        This measures how "in phase" the token is with the current state at each scale.

        Returns gate_amp, gate_phase: (batch, num_scales, K) each in (0, 1).
        """
        # Coherence: how aligned is this token with the current state?
        phase_diff = state.phases - delta_phase  # (B, S, K)
        coherence_per_k = state.amplitudes * delta_amp * torch.cos(phase_diff)  # (B, S, K)
        coherence_per_scale = coherence_per_k.mean(dim=-1)  # (B, S)

        # Gate logit = bias + sensitivity * coherence
        # Expand gate to (B, S, K) — same gate across all K frequencies within a scale
        gate_logit = gate_bias + self.gate_sensitivity * coherence_per_scale  # (B, S)
        gate = torch.sigmoid(gate_logit).unsqueeze(-1)  # (B, S, 1) → broadcast to (B, S, K)

        # Use same gate for amp and phase (could separate later if needed)
        return gate, gate

    @property
    def params_per_token(self) -> int:
        return self.num_scales * self.K * 2 + self.num_scales  # delta_amp + delta_phase + gate_bias


class GatedSpectralStateLM(nn.Module):
    """Bidirectional Gated Spectral State Language Model.

    Like v7 but with content-dependent gating on state updates.
    The gate decides how much of the existing spectral state to retain
    vs overwrite when processing each token.
    """

    def __init__(self, vocab_size: int, num_scales: int = 3, K: int = 4):
        super().__init__()
        self.embedding = GatedSpectralEmbedding(vocab_size, num_scales, K)

        self.combine_weight = nn.Parameter(torch.tensor(0.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _build_spectral_states(
        self,
        token_ids: torch.Tensor,     # (batch, seq_len)
        mask: torch.Tensor,           # (batch, seq_len)
    ) -> tuple[list[SpectralState], list[SpectralState]]:
        """Build forward and backward spectral states with gated updates."""
        B, T = token_ids.shape
        S = self.embedding.num_scales
        K = self.embedding.K
        device = token_ids.device

        # Get all perturbations and gate biases at once
        delta_amp, delta_phase = self.embedding.get_perturbation(token_ids)
        gate_bias = self.embedding.get_gate_bias(token_ids)  # (B, T, S)

        # Forward pass: left to right
        fwd_states = []
        state = SpectralState.zeros(B, S, K, device)
        for t in range(T):
            fwd_states.append(state)
            if not mask[:, t].all():
                active = (~mask[:, t]).float()[:, None, None]  # (B, 1, 1)
                d_amp = delta_amp[:, t] * active
                d_phase = delta_phase[:, t] * active
                g_bias = gate_bias[:, t]  # (B, S)

                # Compute content-dependent gate
                gate_a, gate_p = self.embedding.compute_gate(
                    state, d_amp, d_phase, g_bias
                )

                # Gated update
                state = state.gated_update(d_amp, d_phase, gate_a, gate_p)

        # Backward pass: right to left
        bwd_states = [None] * T
        state = SpectralState.zeros(B, S, K, device)
        for t in range(T - 1, -1, -1):
            bwd_states[t] = state
            if not mask[:, t].all():
                active = (~mask[:, t]).float()[:, None, None]
                d_amp = delta_amp[:, t] * active
                d_phase = delta_phase[:, t] * active
                g_bias = gate_bias[:, t]

                gate_a, gate_p = self.embedding.compute_gate(
                    state, d_amp, d_phase, g_bias
                )
                state = state.gated_update(d_amp, d_phase, gate_a, gate_p)

        return fwd_states, bwd_states

    def _combine_states(self, fwd: SpectralState, bwd: SpectralState) -> SpectralState:
        w = torch.sigmoid(self.combine_weight)
        return SpectralState(
            w * fwd.amplitudes + (1 - w) * bwd.amplitudes,
            w * fwd.phases + (1 - w) * bwd.phases,
        )

    def _score_tokens(self, state: SpectralState) -> torch.Tensor:
        """Score all vocab tokens via cos/sin decomposed matmul (same as v7)."""
        S, K = self.embedding.num_scales, self.embedding.K

        s_amp = state.amplitudes
        s_phase = state.phases
        s_real = (s_amp * torch.cos(s_phase)).view(-1, S * K)
        s_imag = (s_amp * torch.sin(s_phase)).view(-1, S * K)

        v_amp = self.embedding.delta_amp
        v_phase = self.embedding.delta_phase
        v_real = (v_amp * torch.cos(v_phase)).view(-1, S * K)
        v_imag = (v_amp * torch.sin(v_phase)).view(-1, S * K)

        logits = s_real @ v_real.t() + s_imag @ v_imag.t()
        return logits / (self.temperature.abs() + 1e-6)

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits at masked positions."""
        B, T = token_ids.shape

        fwd_states, bwd_states = self._build_spectral_states(token_ids, mask)

        all_logits = []
        for t in range(T):
            if mask[:, t].any():
                combined = self._combine_states(fwd_states[t], bwd_states[t])
                logits = self._score_tokens(combined)
                masked_in_batch = mask[:, t]
                all_logits.append(logits[masked_in_batch])

        if all_logits:
            return torch.cat(all_logits, dim=0)
        else:
            return torch.zeros(0, self.embedding.vocab_size, device=token_ids.device)

    def get_targets(self, token_ids, mask):
        return token_ids[mask]

    def spectral_similarity(self, id1, id2):
        """Same as v7 — compare raw perturbations."""
        d_amp1, d_phase1 = self.embedding.get_perturbation(id1)
        d_amp2, d_phase2 = self.embedding.get_perturbation(id2)

        phase_diff = d_phase1 - d_phase2
        coherence = (d_amp1 * d_amp2 * torch.cos(phase_diff)).sum(dim=(-2, -1))

        norm1 = (d_amp1 ** 2).sum(dim=(-2, -1)).sqrt()
        norm2 = (d_amp2 ** 2).sum(dim=(-2, -1)).sqrt()
        return coherence / (norm1 * norm2 + 1e-8)


def mask_tokens(
    token_ids: torch.Tensor,
    mask_prob: float = 0.15,
    pad_id: int = 0,
    unk_id: int = 1,
) -> torch.Tensor:
    """Create MLM mask. Don't mask special tokens."""
    mask = torch.rand_like(token_ids.float()) < mask_prob
    mask = mask & (token_ids != pad_id) & (token_ids != unk_id)
    return mask
