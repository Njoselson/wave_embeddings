"""Contrastive learning with wave interference energy similarity.

Wave interference energy provides a physically-motivated similarity metric:
when two tokens share frequencies, their waves constructively interfere,
producing higher total energy. When frequencies differ, cross-terms average
to zero and energy is just the sum of individual energies.

Analytically for complex exponentials z1(t) = A1*exp(j*2pi*f1*t) and
z2(t) = A2*exp(j*2pi*f2*t):
    E = A1^2 + A2^2 + 2*A1*A2*sinc(delta_f)

The cross-term 2*A1*A2*sinc(delta_f) is the similarity signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.wave_embedding_v3 import WaveEmbeddingV3, HarmonicWaveEmbedding


def wave_interference_energy(
    f1: torch.Tensor,
    A1: torch.Tensor,
    f2: torch.Tensor,
    A2: torch.Tensor,
    sample_points: int = 256,
) -> torch.Tensor:
    """Compute interference energy between two sets of wave parameters.

    Uses discrete sampling to compute E = (1/T) * sum|z1(t) + z2(t)|^2,
    which captures constructive/destructive interference.

    Args:
        f1: (batch,) or (batch, num_waves) frequencies for token 1
        A1: (batch,) or (batch, num_waves) amplitudes for token 1
        f2: (batch,) or (batch, num_waves) frequencies for token 2
        A2: (batch,) or (batch, num_waves) amplitudes for token 2
        sample_points: number of time samples

    Returns:
        (batch,) interference energy per pair
    """
    # Ensure 2D: (batch, num_waves)
    if f1.dim() == 1:
        f1 = f1.unsqueeze(-1)
        A1 = A1.unsqueeze(-1)
        f2 = f2.unsqueeze(-1)
        A2 = A2.unsqueeze(-1)

    # t: (1, 1, P)
    t = torch.linspace(0, 1, sample_points, device=f1.device, dtype=f1.dtype)
    t = t.view(1, 1, sample_points)

    # phase: (batch, num_waves, P)
    phase1 = 2 * math.pi * f1.unsqueeze(-1) * t
    phase2 = 2 * math.pi * f2.unsqueeze(-1) * t

    # Complex signals: z(t) = A * exp(j * 2pi * f * t)
    real1 = A1.unsqueeze(-1) * torch.cos(phase1)
    imag1 = A1.unsqueeze(-1) * torch.sin(phase1)
    real2 = A2.unsqueeze(-1) * torch.cos(phase2)
    imag2 = A2.unsqueeze(-1) * torch.sin(phase2)

    # Sum over waves -> (batch, P)
    real_sum = real1.sum(dim=1) + real2.sum(dim=1)
    imag_sum = imag1.sum(dim=1) + imag2.sum(dim=1)

    # Energy = mean |z(t)|^2
    energy = (real_sum ** 2 + imag_sum ** 2).mean(dim=-1)  # (batch,)

    return energy


def wave_similarity(
    f1: torch.Tensor,
    A1: torch.Tensor,
    f2: torch.Tensor,
    A2: torch.Tensor,
    sample_points: int = 256,
) -> torch.Tensor:
    """Compute normalized similarity via interference cross-term.

    Returns just the cross-term component, normalized so that
    identical frequencies give similarity ~1 and different frequencies ~0.

    The cross-term is: E_combined - E_self1 - E_self2
    Normalized by: 2 * sqrt(E_self1 * E_self2)
    """
    E_combined = wave_interference_energy(f1, A1, f2, A2, sample_points)

    # Self-energies: sum of A^2 for each wave (analytically)
    if A1.dim() == 1:
        E_self1 = A1 ** 2
        E_self2 = A2 ** 2
    else:
        E_self1 = (A1 ** 2).sum(dim=-1)
        E_self2 = (A2 ** 2).sum(dim=-1)

    cross_term = E_combined - E_self1 - E_self2
    normalizer = 2 * torch.sqrt(E_self1 * E_self2 + 1e-8)

    return cross_term / normalizer


class SkipGramWaveModel(nn.Module):
    """Skip-gram model using wave interference energy for similarity.

    Uses WaveEmbeddingV3 for both target and context embeddings.
    Similarity is computed via wave interference energy.
    """

    def __init__(
        self,
        vocab_size: int,
        num_waves: int = 3,
        sample_points: int = 256,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.target_embedding = WaveEmbeddingV3(
            vocab_size=vocab_size,
            num_waves=num_waves,
        )
        if share_embeddings:
            self.context_embedding = self.target_embedding
        else:
            self.context_embedding = WaveEmbeddingV3(
                vocab_size=vocab_size,
                num_waves=num_waves,
            )
        self.sample_points = sample_points

    def forward(
        self,
        target_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute interference energy for positive and negative pairs.

        Args:
            target_ids: (batch,) target token IDs
            positive_ids: (batch,) positive context token IDs
            negative_ids: (batch, num_negatives) negative token IDs

        Returns:
            pos_energy: (batch,) energy for positive pairs
            neg_energy: (batch, num_negatives) energy for negative pairs
        """
        # Target wave params
        t_f = self.target_embedding.frequencies[target_ids]  # (batch, num_waves)
        t_A = self.target_embedding.amplitudes[target_ids]

        # Positive context params
        p_f = self.context_embedding.frequencies[positive_ids]
        p_A = self.context_embedding.amplitudes[positive_ids]

        pos_energy = wave_interference_energy(
            t_f, t_A, p_f, p_A, self.sample_points
        )

        # Negative context params
        batch_size, num_negatives = negative_ids.shape
        n_f = self.context_embedding.frequencies[negative_ids]  # (batch, neg, waves)
        n_A = self.context_embedding.amplitudes[negative_ids]

        # Expand target for each negative
        t_f_exp = t_f.unsqueeze(1).expand_as(n_f)  # (batch, neg, waves)
        t_A_exp = t_A.unsqueeze(1).expand_as(n_A)

        # Reshape to compute all negative energies at once
        neg_energy = wave_interference_energy(
            t_f_exp.reshape(-1, t_f.size(-1)),
            t_A_exp.reshape(-1, t_A.size(-1)),
            n_f.reshape(-1, n_f.size(-1)),
            n_A.reshape(-1, n_A.size(-1)),
            self.sample_points,
        ).reshape(batch_size, num_negatives)

        return pos_energy, neg_energy


def word_interference_energy_analytical(
    char_ids_1: torch.Tensor,
    char_ids_2: torch.Tensor,
    embedding: WaveEmbeddingV3,
    mask_1: torch.Tensor | None = None,
    mask_2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute interference energy analytically between two words composed of characters.

    Word signal z_W(t) = sum_i sum_k A_{c_i,k} * exp(j*2pi*f_{c_i,k}*t)
    Energy E = E_self1 + E_self2 + E_cross
    E_cross = 2 * sum_{i,j} sum_{k,l} A1_{i,k} * A2_{j,l} * sinc(f1_{i,k} - f2_{j,l})

    Args:
        char_ids_1: (batch, max_len1) character IDs for word 1
        char_ids_2: (batch, max_len2) character IDs for word 2
        embedding: WaveEmbeddingV3 with character-level params
        mask_1: (batch, max_len1) boolean mask, True = valid char
        mask_2: (batch, max_len2) boolean mask, True = valid char

    Returns:
        (batch,) total interference energy per pair
    """
    # Look up params: (batch, max_len, num_waves)
    f1 = embedding.frequencies[char_ids_1]
    A1 = embedding.amplitudes[char_ids_1]
    f2 = embedding.frequencies[char_ids_2]
    A2 = embedding.amplitudes[char_ids_2]

    batch, len1, K = f1.shape
    _, len2, _ = f2.shape

    # Apply masks by zeroing amplitudes of padding chars
    if mask_1 is not None:
        A1 = A1 * mask_1.unsqueeze(-1).float()
    if mask_2 is not None:
        A2 = A2 * mask_2.unsqueeze(-1).float()

    # Flatten chars*waves: (batch, len1*K)
    f1_flat = f1.reshape(batch, -1)
    A1_flat = A1.reshape(batch, -1)
    f2_flat = f2.reshape(batch, -1)
    A2_flat = A2.reshape(batch, -1)

    # Self-energy: sum of A^2 for each word
    E_self1 = (A1_flat ** 2).sum(dim=-1)  # (batch,)
    E_self2 = (A2_flat ** 2).sum(dim=-1)

    # Intra-word cross terms for self-energy (sinc between waves within same word)
    # E_self_full = sum_i sum_j A_i * A_j * sinc(f_i - f_j) for all i,j in word
    # When i==j, sinc(0)=1, giving A_i^2 terms above. Cross terms add interference.
    n1 = len1 * K
    n2 = len2 * K

    # The integral of cos(2π·Δf·t) over [0,1] = sin(2π·Δf)/(2π·Δf) = torch.sinc(2·Δf)
    # since torch.sinc(x) = sin(π·x)/(π·x)

    # Self cross-terms for word 1
    df1_self = f1_flat.unsqueeze(-1) - f1_flat.unsqueeze(-2)  # (batch, n1, n1)
    sinc1_self = torch.sinc(2.0 * df1_self)
    amp1_outer = A1_flat.unsqueeze(-1) * A1_flat.unsqueeze(-2)  # (batch, n1, n1)
    E_self1_full = (amp1_outer * sinc1_self).sum(dim=(-2, -1))

    # Self cross-terms for word 2
    df2_self = f2_flat.unsqueeze(-1) - f2_flat.unsqueeze(-2)  # (batch, n2, n2)
    sinc2_self = torch.sinc(2.0 * df2_self)
    amp2_outer = A2_flat.unsqueeze(-1) * A2_flat.unsqueeze(-2)
    E_self2_full = (amp2_outer * sinc2_self).sum(dim=(-2, -1))

    # Cross-term between word1 and word2
    # delta_f: (batch, n1, n2)
    delta_f = f1_flat.unsqueeze(-1) - f2_flat.unsqueeze(-2)
    sinc_val = torch.sinc(2.0 * delta_f)
    amp_outer = A1_flat.unsqueeze(-1) * A2_flat.unsqueeze(-2)  # (batch, n1, n2)
    E_cross = 2.0 * (amp_outer * sinc_val).sum(dim=(-2, -1))

    return E_self1_full + E_self2_full + E_cross


class CharSkipGramWaveModel(nn.Module):
    """Skip-gram model operating at word level with character-level wave composition.

    Each character has wave parameters. A word's signal is the sum of its
    character complex exponentials. Similarity is computed analytically via
    wave interference energy cross-terms.
    """

    def __init__(self, char_vocab_size: int, num_waves: int = 3):
        super().__init__()
        self.embedding = WaveEmbeddingV3(
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
        """Compute analytical interference energy for positive and negative pairs.

        Args:
            target_chars: (batch, max_len_t) char IDs for target words
            pos_chars: (batch, max_len_p) char IDs for positive context words
            neg_chars: (batch, num_neg, max_len_n) char IDs for negative words
            target_mask: (batch, max_len_t) boolean mask
            pos_mask: (batch, max_len_p) boolean mask
            neg_mask: (batch, num_neg, max_len_n) boolean mask

        Returns:
            pos_energy: (batch,)
            neg_energy: (batch, num_neg)
        """
        pos_energy = word_interference_energy_analytical(
            target_chars, pos_chars, self.embedding, target_mask, pos_mask
        )

        batch_size, num_neg, max_len_n = neg_chars.shape
        max_len_t = target_chars.shape[1]

        # Expand target for each negative
        target_exp = target_chars.unsqueeze(1).expand(batch_size, num_neg, max_len_t)
        target_mask_exp = target_mask.unsqueeze(1).expand(batch_size, num_neg, max_len_t)

        # Reshape to (batch*num_neg, ...)
        neg_energy = word_interference_energy_analytical(
            target_exp.reshape(-1, max_len_t),
            neg_chars.reshape(-1, max_len_n),
            self.embedding,
            target_mask_exp.reshape(-1, max_len_t),
            neg_mask.reshape(-1, max_len_n),
        ).reshape(batch_size, num_neg)

        return pos_energy, neg_energy


# ===== Discrete harmonic composition (v2: fast, supports harmonics + position) =====


def compose_word_signal(
    char_ids: torch.Tensor,
    embedding: HarmonicWaveEmbedding,
    mask: torch.Tensor | None = None,
    sample_points: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a word's complex signal from character wave params with harmonics.

    Each character emits fundamentals + harmonics with position-dependent phase:
      z_word(t) = Σ_i Σ_k Σ_h  (A_{c_i,k} / h^{d_{c_i,k}}) *
                                 exp(j·2π·(h·f_{c_i,k}·t + i·β))

    Args:
        char_ids: (batch, max_len) character IDs
        embedding: HarmonicWaveEmbedding with frequencies, amplitudes, decays, position_freq
        mask: (batch, max_len) boolean mask, True = valid char
        sample_points: P — number of time samples

    Returns:
        real: (batch, P) real part of composed signal
        imag: (batch, P) imaginary part
    """
    # Look up params: (batch, L, K)
    f = embedding.frequencies[char_ids]
    A = embedding.amplitudes[char_ids]
    d = embedding.decays[char_ids]

    if mask is not None:
        A = A * mask.unsqueeze(-1).float()

    batch, L, K = f.shape
    H = embedding.num_harmonics
    P = sample_points

    # Time grid: (P,)
    t = torch.linspace(0, 1, P, device=f.device, dtype=f.dtype)

    # Harmonic indices: (H,)
    h = torch.arange(1, H + 1, device=f.device, dtype=f.dtype)

    # Harmonic amplitudes: A_k / h^d_k
    # A: (batch, L, K) -> (batch, L, K, 1)
    # d: (batch, L, K) -> (batch, L, K, 1)
    # h: (H,) -> (1, 1, 1, H)
    harm_amp = A.unsqueeze(-1) / (h.view(1, 1, 1, H) ** d.unsqueeze(-1))  # (batch, L, K, H)

    # Harmonic frequencies: h * f_k
    # f: (batch, L, K) -> (batch, L, K, 1)
    harm_freq = f.unsqueeze(-1) * h.view(1, 1, 1, H)  # (batch, L, K, H)

    # Position phase: i * β for each character position
    positions = torch.arange(L, device=f.device, dtype=f.dtype)  # (L,)
    pos_phase = positions.view(1, -1, 1, 1) * embedding.position_freq  # (1, L, 1, 1)

    # Total phase at each sample point:
    # 2π * (h*f*t + i*β)
    # harm_freq: (batch, L, K, H), t: (P,) -> phase: (batch, L, K, H, P)
    freq_phase = harm_freq.unsqueeze(-1) * t.view(1, 1, 1, 1, P)  # (batch, L, K, H, P)
    phase = 2 * math.pi * (freq_phase + pos_phase.unsqueeze(-1))  # broadcast pos_phase

    # Complex signal: sum over chars (L), waves (K), harmonics (H) -> (batch, P)
    # harm_amp: (batch, L, K, H) -> (batch, L, K, H, 1)
    real = (harm_amp.unsqueeze(-1) * torch.cos(phase)).sum(dim=(1, 2, 3))  # (batch, P)
    imag = (harm_amp.unsqueeze(-1) * torch.sin(phase)).sum(dim=(1, 2, 3))  # (batch, P)

    return real, imag


def word_energy_discrete(
    real1: torch.Tensor,
    imag1: torch.Tensor,
    real2: torch.Tensor,
    imag2: torch.Tensor,
) -> torch.Tensor:
    """Compute interference energy between two composed word signals.

    E = mean |z1(t) + z2(t)|^2

    Args:
        real1, imag1: (batch, P) composed signal for word 1
        real2, imag2: (batch, P) composed signal for word 2

    Returns:
        (batch,) interference energy
    """
    real_sum = real1 + real2
    imag_sum = imag1 + imag2
    return (real_sum ** 2 + imag_sum ** 2).mean(dim=-1)


class HarmonicCharSkipGramModel(nn.Module):
    """Skip-gram with harmonic wave composition: chars → words via discrete signals.

    Each character has (frequency, amplitude, harmonic_decay) per wave.
    Words are composed by summing character signals (with harmonics + position phase).
    Similarity = interference energy between composed word signals.

    Fast: O((n+m)·K·H·P) per pair, vs O(n·m·K²) for analytical.
    """

    def __init__(
        self,
        char_vocab_size: int,
        num_waves: int = 3,
        num_harmonics: int = 4,
        sample_points: int = 64,
    ):
        super().__init__()
        self.embedding = HarmonicWaveEmbedding(
            vocab_size=char_vocab_size,
            num_waves=num_waves,
            num_harmonics=num_harmonics,
        )
        self.sample_points = sample_points

    def compose(
        self,
        char_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose word signal from character IDs."""
        return compose_word_signal(char_ids, self.embedding, mask, self.sample_points)

    def forward(
        self,
        target_chars: torch.Tensor,
        pos_chars: torch.Tensor,
        neg_chars: torch.Tensor,
        target_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute interference energy for positive and negative word pairs.

        Args:
            target_chars: (batch, max_len_t) char IDs
            pos_chars: (batch, max_len_p) char IDs
            neg_chars: (batch, num_neg, max_len_n) char IDs
            target_mask, pos_mask: (batch, max_len) boolean
            neg_mask: (batch, num_neg, max_len_n) boolean

        Returns:
            pos_energy: (batch,)
            neg_energy: (batch, num_neg)
        """
        # Compose target and positive signals
        t_real, t_imag = self.compose(target_chars, target_mask)  # (batch, P)
        p_real, p_imag = self.compose(pos_chars, pos_mask)

        pos_energy = word_energy_discrete(t_real, t_imag, p_real, p_imag)

        # Compose negative signals
        batch_size, num_neg, max_len_n = neg_chars.shape
        # Flatten negatives to (batch*num_neg, max_len_n)
        n_real, n_imag = self.compose(
            neg_chars.reshape(-1, max_len_n),
            neg_mask.reshape(-1, max_len_n),
        )  # (batch*num_neg, P)

        # Expand target signal to match negatives
        t_real_exp = t_real.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, t_real.shape[-1])
        t_imag_exp = t_imag.unsqueeze(1).expand(-1, num_neg, -1).reshape(-1, t_imag.shape[-1])

        neg_energy = word_energy_discrete(
            t_real_exp, t_imag_exp, n_real, n_imag
        ).reshape(batch_size, num_neg)

        return pos_energy, neg_energy


def negative_sampling_loss(
    pos_energy: torch.Tensor,
    neg_energy: torch.Tensor,
) -> torch.Tensor:
    """Negative sampling loss using interference energy as score.

    Maximizes energy for positive pairs, minimizes for negative pairs.

    Args:
        pos_energy: (batch,) energy for positive pairs
        neg_energy: (batch, num_negatives) energy for negative pairs

    Returns:
        scalar loss
    """
    pos_loss = -F.logsigmoid(pos_energy).mean()
    neg_loss = -F.logsigmoid(-neg_energy).mean()
    return pos_loss + neg_loss
