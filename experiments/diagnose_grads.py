"""Diagnose why frequency parameters aren't learning."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.wave_embedding import WaveTokenEmbedding


def diagnose():
    torch.manual_seed(42)
    emb = WaveTokenEmbedding(vocab_size=100, num_waves=7, signal_length=256, k_max=8)

    token_ids = torch.tensor([[0, 1, 2, 3, 4]])

    # Forward pass
    out = emb(token_ids)
    loss = out.sum()
    loss.backward()

    print("=== Gradient magnitudes ===")
    print(f"Frequencies grad norm: {emb.frequencies.grad.norm():.6f}")
    print(f"Amplitudes grad norm:  {emb.amplitudes.grad.norm():.6f}")
    print(f"Harmonics grad norm:   {emb.harmonics.grad.norm():.6f}")

    print(f"\nFreq grad mean abs:    {emb.frequencies.grad.abs().mean():.6f}")
    print(f"Amp grad mean abs:     {emb.amplitudes.grad.abs().mean():.6f}")
    print(f"Harm grad mean abs:    {emb.harmonics.grad.abs().mean():.6f}")

    # Check ratio - how much smaller are freq grads vs amp grads?
    ratio = emb.amplitudes.grad.abs().mean() / emb.frequencies.grad.abs().mean()
    print(f"\nAmp/Freq grad ratio:   {ratio:.1f}x")

    # Check the spectrum values
    print(f"\n=== Spectrum stats ===")
    with torch.no_grad():
        out2 = emb(token_ids)
        print(f"Spectrum mean: {out2.mean():.4f}")
        print(f"Spectrum std:  {out2.std():.4f}")
        print(f"Spectrum max:  {out2.max():.4f}")
        print(f"Spectrum min:  {out2.min():.4f}")

    # Test: what if we use real+imag instead of abs?
    print("\n=== Comparing abs() vs real+imag gradients ===")
    emb2 = WaveTokenEmbedding(vocab_size=100, num_waves=7, signal_length=256, k_max=8)
    # Monkey-patch to use real+imag
    original_forward = emb2.forward

    def forward_realimag(token_ids):
        batch_size, seq_len = token_ids.shape
        flat_ids = token_ids.view(-1)
        f, A, H = emb2.get_wave_params(flat_ids)
        signals = emb2.tone_wave(f, A, H)
        composite = signals.sum(dim=1)
        spectrum = torch.fft.rfft(composite, dim=-1)
        # Use real and imag stacked instead of abs
        embedding = torch.cat([spectrum.real, spectrum.imag], dim=-1)
        embedding = embedding.view(batch_size, seq_len, -1)
        return embedding

    out3 = forward_realimag(token_ids)
    loss3 = out3.sum()
    loss3.backward()

    print(f"Freq grad norm (real+imag): {emb2.frequencies.grad.norm():.6f}")
    print(f"Amp grad norm (real+imag):  {emb2.amplitudes.grad.norm():.6f}")
    print(f"Harm grad norm (real+imag): {emb2.harmonics.grad.norm():.6f}")

    ratio2 = emb2.amplitudes.grad.abs().mean() / emb2.frequencies.grad.abs().mean()
    print(f"Amp/Freq grad ratio:        {ratio2:.1f}x")


if __name__ == "__main__":
    diagnose()
