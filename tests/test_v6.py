"""Tests for v6: multi-scale resonance with phase, decay, gating."""

import torch
import torch.nn.functional as F
import pytest

from src.wave_embedding_v6 import (
    WaveEmbeddingV6,
    SkipGramV6,
    WaveLMv6,
    energy,
    energy_no_phase,
    self_energy,
    similarity,
    negative_sampling_loss,
    freq_diversity_loss,
)


class TestWaveEmbeddingV6:
    def test_params_per_token(self):
        """Each token should have 5 params: freq_slow, freq_fast, amplitude, phase, scale_mix."""
        emb = WaveEmbeddingV6(vocab_size=100, num_harmonics=7)
        assert emb.freq_slow.shape == (100,)
        assert emb.freq_fast.shape == (100,)
        assert emb.amplitudes.shape == (100,)
        assert emb.phase.shape == (100,)
        assert emb.scale_mix.shape == (100,)

    def test_total_params(self):
        """Total params = 5 * vocab_size + 2 (decay_slow, decay_fast)."""
        emb = WaveEmbeddingV6(vocab_size=100, num_harmonics=7)
        total = sum(p.numel() for p in emb.parameters())
        assert total == 502  # 5 * 100 + 2

    def test_harmonics_shape(self):
        """get_harmonics should return (..., 2*H) tensors for dual-band."""
        emb = WaveEmbeddingV6(vocab_size=100, num_harmonics=7)
        ids = torch.tensor([0, 5, 10])
        f, A, phi = emb.get_harmonics(ids)
        assert f.shape == (3, 14)  # 2 * 7
        assert A.shape == (3, 14)
        assert phi.shape == (3, 14)

    def test_slow_fast_separation(self):
        """Slow band should use freq_slow, fast band should use freq_fast."""
        emb = WaveEmbeddingV6(vocab_size=10, num_harmonics=4)
        emb.freq_slow.data = torch.tensor([0.5] * 10)
        emb.freq_fast.data = torch.tensor([5.0] * 10)

        ids = torch.tensor([0])
        f, _, _ = emb.get_harmonics(ids)

        # First H harmonics are slow band: [0.5, 1.0, 1.5, 2.0]
        assert torch.allclose(f[0, :4], torch.tensor([0.5, 1.0, 1.5, 2.0]))
        # Last H harmonics are fast band: [5.0, 10.0, 15.0, 20.0]
        assert torch.allclose(f[0, 4:], torch.tensor([5.0, 10.0, 15.0, 20.0]))

    def test_scale_mix_controls_allocation(self):
        """scale_mix=large → most energy in slow band; scale_mix=-large → fast band."""
        emb = WaveEmbeddingV6(vocab_size=10, num_harmonics=4)
        emb.amplitudes.data = torch.ones(10)

        # Set scale_mix very positive → sigmoid ≈ 1 → slow gets most energy
        emb.scale_mix.data = torch.tensor([10.0] * 10)
        ids = torch.tensor([0])
        _, A, _ = emb.get_harmonics(ids)
        slow_energy = A[0, :4].pow(2).sum()
        fast_energy = A[0, 4:].pow(2).sum()
        assert slow_energy > fast_energy * 10

        # Set scale_mix very negative → sigmoid ≈ 0 → fast gets most energy
        emb.scale_mix.data = torch.tensor([-10.0] * 10)
        _, A2, _ = emb.get_harmonics(ids)
        slow_energy2 = A2[0, :4].pow(2).sum()
        fast_energy2 = A2[0, 4:].pow(2).sum()
        assert fast_energy2 > slow_energy2 * 10

    def test_phase_initialization(self):
        """Phase should initialize to zero."""
        emb = WaveEmbeddingV6(vocab_size=100)
        assert (emb.phase.data == 0).all()


class TestPhaseAwareEnergy:
    def test_phase_zero_matches_no_phase(self):
        """With phase=0 everywhere, energy should match v5-style energy_no_phase."""
        torch.manual_seed(42)
        f1 = torch.randn(4, 14)
        A1 = torch.rand(4, 14) + 0.5
        f2 = torch.randn(4, 14)
        A2 = torch.rand(4, 14) + 0.5
        phi_zero = torch.zeros(4, 14)

        e_phased = energy(f1, A1, phi_zero, f2, A2, phi_zero)
        e_nophase = energy_no_phase(f1, A1, f2, A2)
        assert torch.allclose(e_phased, e_nophase, atol=1e-5), \
            f"phase=0 energy should match no-phase: {e_phased} vs {e_nophase}"

    def test_phase_shift_changes_energy(self):
        """Shifting phase of one token should change cross-term energy."""
        f = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        A = torch.tensor([[1.0, 0.5, 0.33, 0.25]])
        phi_zero = torch.zeros(1, 4)
        phi_shift = torch.full((1, 4), 1.5)

        e_same = energy(f, A, phi_zero, f, A, phi_zero)
        e_shifted = energy(f, A, phi_zero, f, A, phi_shift)
        assert not torch.allclose(e_same, e_shifted, atol=1e-3), \
            "Phase shift should change energy"

    def test_energy_symmetric(self):
        """Energy should be symmetric: E(a,b) = E(b,a)."""
        torch.manual_seed(42)
        f1, A1, phi1 = torch.randn(4, 14), torch.rand(4, 14) + 0.5, torch.randn(4, 14) * 0.5
        f2, A2, phi2 = torch.randn(4, 14), torch.rand(4, 14) + 0.5, torch.randn(4, 14) * 0.5
        assert torch.allclose(
            energy(f1, A1, phi1, f2, A2, phi2),
            energy(f2, A2, phi2, f1, A1, phi1),
            atol=1e-5,
        )

    def test_self_similarity_near_one(self):
        """Self-similarity should be close to 1."""
        f = torch.tensor([[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]])
        A = torch.tensor([[1.0, 0.5, 0.33, 1.0, 0.5, 0.33]])
        phi = torch.zeros(1, 6)
        s = similarity(f, A, phi, f, A, phi)
        assert abs(s.item() - 1.0) < 0.05


class TestFreqDiversityLoss:
    def test_identical_freqs_high_loss(self):
        """All same frequency → high diversity loss."""
        freqs = torch.ones(100)
        loss = freq_diversity_loss(freqs, margin=0.1)
        assert loss.item() > 0.05

    def test_spread_freqs_low_loss(self):
        """Well-spread frequencies → low diversity loss."""
        freqs = torch.linspace(-10, 10, 100)
        loss = freq_diversity_loss(freqs, margin=0.1)
        assert loss.item() < 0.01


class TestSkipGramV6:
    def test_forward_shapes(self):
        model = SkipGramV6(vocab_size=100, num_harmonics=7)
        target = torch.randint(0, 100, (8,))
        pos = torch.randint(0, 100, (8,))
        neg = torch.randint(0, 100, (8, 5))
        pos_e, neg_e = model(target, pos, neg)
        assert pos_e.shape == (8,)
        assert neg_e.shape == (8, 5)

    def test_gradients_flow_all_params(self):
        """All 5 per-token params + 2 global params should receive gradients."""
        model = SkipGramV6(vocab_size=50, num_harmonics=7)
        # Init phase slightly off-zero so cos(dphi) gradient is nonzero
        # (at dphi=0, d/dphi cos(dphi) = -sin(0) = 0)
        model.embedding.phase.data = torch.randn(50) * 0.1
        target = torch.randint(0, 50, (8,))
        pos = torch.randint(0, 50, (8,))
        neg = torch.randint(0, 50, (8, 5))
        pos_e, neg_e = model(target, pos, neg)
        loss = negative_sampling_loss(pos_e, neg_e)
        loss.backward()

        # Per-token params
        assert model.embedding.freq_slow.grad is not None
        assert model.embedding.freq_fast.grad is not None
        assert model.embedding.amplitudes.grad is not None
        assert model.embedding.phase.grad is not None
        assert model.embedding.scale_mix.grad is not None

        # Global params
        assert model.embedding.decay_slow.grad is not None
        assert model.embedding.decay_fast.grad is not None

        # Non-zero gradients
        assert model.embedding.freq_slow.grad.abs().sum() > 0
        assert model.embedding.freq_fast.grad.abs().sum() > 0
        assert model.embedding.phase.grad.abs().sum() > 0

    def test_loss_decreases(self):
        """Skip-gram loss should decrease on identical target/positive."""
        model = SkipGramV6(vocab_size=30, num_harmonics=7)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        target = torch.randint(2, 30, (16,))
        pos = target.clone()
        neg = torch.randint(2, 30, (16, 5))

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pos_e, neg_e = model(target, pos, neg)
            loss = negative_sampling_loss(pos_e, neg_e)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


class TestWaveLMv6:
    def test_logits_shape(self):
        model = WaveLMv6(vocab_size=50, num_harmonics=4)
        ids = torch.randint(0, 50, (2, 8))
        logits = model(ids, chunk_size=4)
        assert logits.shape == (2, 8, 50)

    def test_first_position_zero(self):
        """Position 0 has no history, so logits should be all zeros (before gating/temp)."""
        model = WaveLMv6(vocab_size=50, num_harmonics=4)
        # Set gate to identity (large gate_filter)
        model.gate_filter.data = torch.ones(model.GATE_SIZE) * 10
        model.gate_bias.data = torch.zeros(model.GATE_SIZE)
        model.temp.data = torch.tensor(1.0)
        ids = torch.randint(0, 50, (2, 8))
        logits = model(ids, chunk_size=4)
        # After gating (≈identity when gate≈1), position 0 should be ≈ 0
        assert logits[:, 0, :].abs().max() < 0.01

    def test_causal_masking(self):
        """Logits at position t should only depend on tokens 0..t-1."""
        model = WaveLMv6(vocab_size=50, num_harmonics=4)
        ids1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        ids2 = torch.tensor([[1, 2, 3, 4, 40, 41, 42, 43]])
        with torch.no_grad():
            logits1 = model(ids1, chunk_size=4)
            logits2 = model(ids2, chunk_size=4)
        # Positions 0-4 should have identical logits (same history)
        assert torch.allclose(logits1[:, :5, :], logits2[:, :5, :], atol=1e-5)
        # Position 5+ should differ
        assert not torch.allclose(logits1[:, 5, :], logits2[:, 5, :])

    def test_decay_reduces_old_contributions(self):
        """With high lambda, old tokens should contribute less than with low lambda."""
        model = WaveLMv6(vocab_size=50, num_harmonics=4)
        ids = torch.randint(2, 50, (1, 16))

        # High decay
        model.lambda_slow_raw.data = torch.tensor(5.0)
        model.lambda_fast_raw.data = torch.tensor(5.0)
        with torch.no_grad():
            logits_high_decay = model(ids, chunk_size=16)

        # Low decay (near zero lambda)
        model.lambda_slow_raw.data = torch.tensor(-10.0)  # softplus → ~0
        model.lambda_fast_raw.data = torch.tensor(-10.0)
        with torch.no_grad():
            logits_low_decay = model(ids, chunk_size=16)

        # With high decay, later positions should have smaller logit magnitudes
        # (old tokens fade away)
        high_mag = logits_high_decay[:, -1, :].abs().mean()
        low_mag = logits_low_decay[:, -1, :].abs().mean()
        assert high_mag < low_mag, \
            f"High decay should reduce magnitudes: {high_mag:.4f} vs {low_mag:.4f}"

    def test_gate_identity(self):
        """When gate_filter is very large (sigmoid→1), gating should be near identity."""
        model = WaveLMv6(vocab_size=30, num_harmonics=4)
        ids = torch.randint(0, 30, (2, 8))

        # Set gate to pass-through
        model.gate_filter.data = torch.ones(model.GATE_SIZE) * 100
        model.gate_bias.data = torch.zeros(model.GATE_SIZE)
        with torch.no_grad():
            logits_gated = model(ids, chunk_size=4)

        # The logits should be non-trivial (not all zero beyond pos 0)
        assert logits_gated[:, 2:, :].abs().max() > 0.01

    def test_gradients_flow(self):
        """All parameters should receive gradients."""
        model = WaveLMv6(vocab_size=30, num_harmonics=4)
        ids = torch.randint(2, 30, (4, 8))
        logits = model(ids, chunk_size=4)
        loss = F.cross_entropy(
            logits[:, 1:, :].reshape(-1, 30),
            ids[:, 1:].reshape(-1),
        )
        loss.backward()

        # Per-token params
        assert model.embedding.freq_slow.grad is not None
        assert model.embedding.freq_fast.grad is not None
        assert model.embedding.amplitudes.grad is not None
        assert model.embedding.phase.grad is not None
        assert model.embedding.scale_mix.grad is not None
        assert model.embedding.freq_slow.grad.abs().sum() > 0

        # Global params
        assert model.embedding.decay_slow.grad is not None
        assert model.embedding.decay_fast.grad is not None
        assert model.lambda_slow_raw.grad is not None
        assert model.lambda_fast_raw.grad is not None
        assert model.gate_filter.grad is not None
        assert model.gate_bias.grad is not None
        assert model.temp.grad is not None

    def test_loss_decreases(self):
        """LM loss should decrease with training on a repeating pattern."""
        torch.manual_seed(42)
        model = WaveLMv6(vocab_size=20, num_harmonics=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        ids = torch.tensor([[2, 3, 4, 5, 2, 3, 4, 5]] * 4)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            logits = model(ids, chunk_size=4)
            loss = F.cross_entropy(
                logits[:, 1:, :].reshape(-1, 20),
                ids[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
