"""Tests for v4: simplified analytical wave embeddings."""

import torch
import pytest

from src.wave_embedding_v4 import (
    WaveEmbeddingV4,
    SkipGramV4,
    energy,
    self_energy,
    similarity,
    negative_sampling_loss,
)


class TestAnalyticalEnergy:
    def test_same_waves_high_energy(self):
        """Identical waves should have high energy (constructive interference)."""
        f = torch.tensor([[1.0, 2.0, 3.0]])
        A = torch.tensor([[1.0, 1.0, 1.0]])
        e = energy(f, A, f, A)
        e_self = self_energy(f, A)
        # Combined energy should be ~4x self energy (amplitude doubles → energy quadruples)
        assert e.item() > 3 * e_self.item()

    def test_different_waves_lower_energy(self):
        """Different frequencies should have lower cross-term (less interference)."""
        f1 = torch.tensor([[1.0, 2.0, 3.0]])
        A1 = torch.tensor([[1.0, 1.0, 1.0]])
        f2 = torch.tensor([[10.0, 20.0, 30.0]])
        A2 = torch.tensor([[1.0, 1.0, 1.0]])
        e_same = energy(f1, A1, f1, A1)
        e_diff = energy(f1, A1, f2, A2)
        assert e_same.item() > e_diff.item()

    def test_energy_symmetric(self):
        """E(a, b) == E(b, a)."""
        f1 = torch.randn(4, 3)
        A1 = torch.rand(4, 3) + 0.5
        f2 = torch.randn(4, 3)
        A2 = torch.rand(4, 3) + 0.5
        e_ab = energy(f1, A1, f2, A2)
        e_ba = energy(f2, A2, f1, A1)
        assert torch.allclose(e_ab, e_ba, atol=1e-5)

    def test_energy_non_negative(self):
        """Energy should always be non-negative."""
        f1 = torch.randn(10, 3)
        A1 = torch.rand(10, 3) + 0.1
        f2 = torch.randn(10, 3)
        A2 = torch.rand(10, 3) + 0.1
        e = energy(f1, A1, f2, A2)
        assert (e >= -1e-6).all()

    def test_self_energy_positive(self):
        """Self-energy should be positive."""
        f = torch.randn(5, 3)
        A = torch.rand(5, 3) + 0.5
        e = self_energy(f, A)
        assert (e > 0).all()

    def test_energy_batched(self):
        """Energy should work with batched inputs."""
        f1 = torch.randn(8, 3)
        A1 = torch.rand(8, 3) + 0.5
        f2 = torch.randn(8, 3)
        A2 = torch.rand(8, 3) + 0.5
        e = energy(f1, A1, f2, A2)
        assert e.shape == (8,)


class TestSimilarity:
    def test_self_similarity_near_one(self):
        """Similarity of a wave set with itself should be ~1."""
        f = torch.tensor([[1.0, 2.0, 3.0]])
        A = torch.tensor([[1.0, 1.0, 1.0]])
        s = similarity(f, A, f, A)
        assert abs(s.item() - 1.0) < 0.05

    def test_different_similarity_lower(self):
        """Well-separated frequencies should have low similarity."""
        f1 = torch.tensor([[1.0, 2.0, 3.0]])
        A1 = torch.tensor([[1.0, 1.0, 1.0]])
        f2 = torch.tensor([[10.0, 20.0, 30.0]])
        A2 = torch.tensor([[1.0, 1.0, 1.0]])
        s = similarity(f1, A1, f2, A2)
        assert s.item() < 0.5


class TestWordComposition:
    def test_position_breaks_commutativity(self):
        """'cat' and 'act' should have different word params."""
        emb = WaveEmbeddingV4(vocab_size=10, num_waves=3)
        # Manually set position_freq to something non-trivial
        emb.position_freq.data = torch.tensor(0.5)

        cat = torch.tensor([[1, 2, 3]])
        act = torch.tensor([[2, 1, 3]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        f_cat, A_cat = emb.get_word_params(cat, mask)
        f_act, A_act = emb.get_word_params(act, mask)

        # Frequencies should differ due to position shifts
        assert not torch.allclose(f_cat, f_act)

    def test_mask_zeros_padding(self):
        """Masked positions should have zero amplitude."""
        emb = WaveEmbeddingV4(vocab_size=10, num_waves=3)
        ids = torch.tensor([[1, 2, 0]])
        mask = torch.tensor([[True, True, False]])

        f, A = emb.get_word_params(ids, mask)
        # Last 3 amplitudes (from masked char) should be zero
        assert (A[0, -3:] == 0).all()
        # First 6 should be non-zero
        assert (A[0, :6] != 0).all()

    def test_word_params_shape(self):
        """Word params should be (batch, num_chars * num_waves)."""
        emb = WaveEmbeddingV4(vocab_size=10, num_waves=3)
        ids = torch.tensor([[1, 2, 3, 4]])  # 4 chars
        mask = torch.ones(1, 4, dtype=torch.bool)
        f, A = emb.get_word_params(ids, mask)
        assert f.shape == (1, 12)  # 4 chars × 3 waves
        assert A.shape == (1, 12)


class TestSkipGramV4:
    def test_forward_shapes(self):
        """Forward should return correct energy shapes."""
        model = SkipGramV4(char_vocab_size=50, num_waves=3)
        batch, num_neg, max_len = 4, 5, 6
        target = torch.randint(0, 50, (batch, max_len))
        pos = torch.randint(0, 50, (batch, max_len))
        neg = torch.randint(0, 50, (batch, num_neg, max_len))
        t_mask = torch.ones(batch, max_len, dtype=torch.bool)
        p_mask = torch.ones(batch, max_len, dtype=torch.bool)
        n_mask = torch.ones(batch, num_neg, max_len, dtype=torch.bool)

        pos_e, neg_e = model(target, pos, neg, t_mask, p_mask, n_mask)
        assert pos_e.shape == (batch,)
        assert neg_e.shape == (batch, num_neg)

    def test_gradients_flow(self):
        """Gradients should reach all parameters."""
        model = SkipGramV4(char_vocab_size=50, num_waves=3)
        batch, num_neg, max_len = 4, 5, 6
        target = torch.randint(0, 50, (batch, max_len))
        pos = torch.randint(0, 50, (batch, max_len))
        neg = torch.randint(0, 50, (batch, num_neg, max_len))
        t_mask = torch.ones(batch, max_len, dtype=torch.bool)
        p_mask = torch.ones(batch, max_len, dtype=torch.bool)
        n_mask = torch.ones(batch, num_neg, max_len, dtype=torch.bool)

        pos_e, neg_e = model(target, pos, neg, t_mask, p_mask, n_mask)
        loss = negative_sampling_loss(pos_e, neg_e)
        loss.backward()

        assert model.embedding.frequencies.grad is not None
        assert model.embedding.amplitudes.grad is not None
        assert model.embedding.position_freq.grad is not None
        assert model.embedding.frequencies.grad.abs().sum() > 0
        assert model.embedding.amplitudes.grad.abs().sum() > 0

    def test_loss_decreases(self):
        """Loss should decrease over a few training steps."""
        model = SkipGramV4(char_vocab_size=30, num_waves=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch, num_neg, max_len = 16, 3, 5
        target = torch.randint(2, 30, (batch, max_len))
        pos = target.clone()  # same as target → should have high energy
        neg = torch.randint(2, 30, (batch, num_neg, max_len))
        t_mask = torch.ones(batch, max_len, dtype=torch.bool)
        p_mask = torch.ones(batch, max_len, dtype=torch.bool)
        n_mask = torch.ones(batch, num_neg, max_len, dtype=torch.bool)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            pos_e, neg_e = model(target, pos, neg, t_mask, p_mask, n_mask)
            loss = negative_sampling_loss(pos_e, neg_e)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_materialize_shape(self):
        """Materialize should produce (batch, dim*2) vectors."""
        model = SkipGramV4(char_vocab_size=50, num_waves=3)
        ids = torch.randint(0, 50, (4, 3))
        vec = model.materialize(ids, dim=32)
        assert vec.shape == (4, 64)  # 32 real + 32 imag


class TestNegativeSamplingLoss:
    def test_loss_positive(self):
        """Loss should be positive."""
        pos_e = torch.randn(8)
        neg_e = torch.randn(8, 5)
        loss = negative_sampling_loss(pos_e, neg_e)
        assert loss.item() > 0

    def test_good_pairs_lower_loss(self):
        """High pos energy + low neg energy should give lower loss."""
        good_loss = negative_sampling_loss(
            torch.tensor([5.0, 5.0]),
            torch.tensor([[-5.0, -5.0], [-5.0, -5.0]]),
        )
        bad_loss = negative_sampling_loss(
            torch.tensor([-5.0, -5.0]),
            torch.tensor([[5.0, 5.0], [5.0, 5.0]]),
        )
        assert good_loss.item() < bad_loss.item()


class TestVsDiscreteSampling:
    """Verify analytical energy matches discrete sampling (the v3 approach)."""

    def test_analytical_matches_discrete(self):
        """Analytical sinc energy should match high-resolution discrete sampling."""
        import math

        f1 = torch.tensor([[1.5, 3.0, 5.0]])
        A1 = torch.tensor([[1.0, 0.8, 0.6]])
        f2 = torch.tensor([[1.5, 4.0, 7.0]])
        A2 = torch.tensor([[0.9, 1.0, 0.5]])

        # Analytical
        e_analytical = energy(f1, A1, f2, A2)

        # Discrete (high resolution for accuracy)
        P = 10000
        t = torch.linspace(0, 1, P).view(1, 1, P)

        phase1 = 2 * math.pi * f1.unsqueeze(-1) * t
        phase2 = 2 * math.pi * f2.unsqueeze(-1) * t
        real1 = (A1.unsqueeze(-1) * torch.cos(phase1)).sum(dim=1)
        imag1 = (A1.unsqueeze(-1) * torch.sin(phase1)).sum(dim=1)
        real2 = (A2.unsqueeze(-1) * torch.cos(phase2)).sum(dim=1)
        imag2 = (A2.unsqueeze(-1) * torch.sin(phase2)).sum(dim=1)
        real_sum = real1 + real2
        imag_sum = imag1 + imag2
        e_discrete = (real_sum ** 2 + imag_sum ** 2).mean(dim=-1)

        assert torch.allclose(e_analytical, e_discrete, atol=0.05), \
            f"analytical={e_analytical.item():.4f} vs discrete={e_discrete.item():.4f}"
