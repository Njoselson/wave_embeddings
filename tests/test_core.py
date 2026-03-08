"""Core tests: gradient flow, signal generation, and end-to-end forward/backward."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tone_wave import ToneWave
from src.wave_embedding import WaveTokenEmbedding
from src.wave_model import WaveEmbeddingModel


class TestToneWave:
    def test_output_shape(self):
        tw = ToneWave(signal_length=512, k_max=8)
        batch, num_waves = 4, 7
        f = torch.rand(batch, num_waves) * 10 + 1
        A = torch.ones(batch, num_waves)
        H = torch.full((batch, num_waves), 4.0)

        signal = tw(f, A, H)
        assert signal.shape == (batch, num_waves, 512)

    def test_amplitude_scaling(self):
        tw = ToneWave(signal_length=512, k_max=8)
        f = torch.tensor([[5.0]])
        H = torch.tensor([[4.0]])

        sig_small = tw(f, torch.tensor([[1.0]]), H)
        sig_big = tw(f, torch.tensor([[3.0]]), H)

        # Signal with 3x amplitude should have ~3x magnitude
        ratio = sig_big.abs().mean() / sig_small.abs().mean()
        assert abs(ratio.item() - 3.0) < 0.01

    def test_zero_amplitude_gives_silence(self):
        tw = ToneWave(signal_length=512, k_max=8)
        f = torch.tensor([[5.0]])
        A = torch.tensor([[0.0]])
        H = torch.tensor([[4.0]])

        signal = tw(f, A, H)
        assert signal.abs().max().item() < 1e-7


class TestWaveTokenEmbedding:
    def test_output_shape(self):
        emb = WaveTokenEmbedding(vocab_size=100, num_waves=7, signal_length=512)
        token_ids = torch.randint(0, 100, (4, 10))

        out = emb(token_ids)
        assert out.shape == (4, 10, 514)  # (512//2 + 1) * 2 for real+imag

    def test_different_tokens_different_embeddings(self):
        emb = WaveTokenEmbedding(vocab_size=100, num_waves=7, signal_length=512)
        ids_a = torch.tensor([[0, 1, 2]])
        ids_b = torch.tensor([[3, 4, 5]])

        out_a = emb(ids_a)
        out_b = emb(ids_b)

        # Different tokens should produce different spectra
        assert not torch.allclose(out_a, out_b)

    def test_same_tokens_same_embeddings(self):
        emb = WaveTokenEmbedding(vocab_size=100, num_waves=7, signal_length=512)
        ids = torch.tensor([[5, 10, 15]])

        out1 = emb(ids)
        out2 = emb(ids)
        assert torch.allclose(out1, out2)


class TestGradientFlow:
    """Critical tests: verify backprop flows through FFT to wave parameters."""

    def test_frequency_gradients(self):
        emb = WaveTokenEmbedding(vocab_size=10, num_waves=3, signal_length=256)
        token_ids = torch.tensor([[0, 1, 2]])

        out = emb(token_ids)
        loss = out.sum()
        loss.backward()

        assert emb.frequencies.grad is not None
        assert emb.frequencies.grad.abs().sum() > 0, "Frequency gradients are all zero"

    def test_amplitude_gradients(self):
        emb = WaveTokenEmbedding(vocab_size=10, num_waves=3, signal_length=256)
        token_ids = torch.tensor([[0, 1, 2]])

        out = emb(token_ids)
        loss = out.sum()
        loss.backward()

        assert emb.amplitudes.grad is not None
        assert emb.amplitudes.grad.abs().sum() > 0, "Amplitude gradients are all zero"

    def test_harmonic_gradients(self):
        emb = WaveTokenEmbedding(vocab_size=10, num_waves=3, signal_length=256)
        token_ids = torch.tensor([[0, 1, 2]])

        out = emb(token_ids)
        loss = out.sum()
        loss.backward()

        assert emb.harmonics.grad is not None
        assert emb.harmonics.grad.abs().sum() > 0, "Harmonic gradients are all zero"

    def test_end_to_end_backward(self):
        model = WaveEmbeddingModel(
            vocab_size=50, num_classes=5, num_waves=7,
            signal_length=256, k_max=8, hidden_dim=64,
        )
        token_ids = torch.randint(0, 50, (8, 12))
        labels = torch.randint(0, 5, (8,))

        logits = model(token_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check all wave params received gradients
        we = model.wave_embedding
        for name, param in [("freq", we.frequencies), ("amp", we.amplitudes), ("harm", we.harmonics)]:
            assert param.grad is not None, f"{name} grad is None"
            assert param.grad.abs().sum() > 0, f"{name} gradients are all zero"

        # Check classifier also got gradients
        for p in model.classifier.parameters():
            assert p.grad is not None

    def test_gradient_stability(self):
        """Check that gradients aren't exploding after a few steps."""
        model = WaveEmbeddingModel(
            vocab_size=50, num_classes=5, num_waves=7,
            signal_length=256, k_max=8, hidden_dim=64,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        grad_norms = []
        for _ in range(10):
            token_ids = torch.randint(0, 50, (8, 12))
            labels = torch.randint(0, 5, (8,))

            optimizer.zero_grad()
            logits = model(token_ids)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()

            total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            grad_norms.append(total_norm)
            optimizer.step()

        # Gradients should not explode (no norm > 100x the first)
        assert max(grad_norms) < grad_norms[0] * 100, f"Gradient explosion detected: {grad_norms}"


class TestWaveModel:
    def test_output_shape(self):
        model = WaveEmbeddingModel(
            vocab_size=100, num_classes=3, num_waves=7,
            signal_length=512, k_max=8, hidden_dim=128,
        )
        token_ids = torch.randint(0, 100, (4, 20))

        logits = model(token_ids)
        assert logits.shape == (4, 3)

    def test_loss_decreases(self):
        """Sanity check: loss should decrease when overfitting a tiny batch."""
        torch.manual_seed(42)
        model = WaveEmbeddingModel(
            vocab_size=20, num_classes=3, num_waves=7,
            signal_length=256, k_max=8, hidden_dim=64,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed tiny batch
        token_ids = torch.randint(0, 20, (4, 8))
        labels = torch.randint(0, 3, (4,))

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(token_ids)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
