"""Diagnose why freq_std stays constant despite gradients existing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.wave_model import WaveEmbeddingModel


def diagnose():
    torch.manual_seed(42)
    model = WaveEmbeddingModel(vocab_size=100, num_classes=2, num_waves=7,
                                signal_length=256, k_max=8, hidden_dim=64)

    we = model.wave_embedding
    print(f"Initial freq stats: mean={we.frequencies.data.mean():.4f} std={we.frequencies.data.std():.4f}")
    print(f"Initial freq range: [{we.frequencies.data.min():.4f}, {we.frequencies.data.max():.4f}]")
    print(f"Initial amp stats:  mean={we.amplitudes.data.mean():.4f} std={we.amplitudes.data.std():.4f}")
    print(f"Initial harm stats: mean={we.harmonics.data.mean():.4f} std={we.harmonics.data.std():.4f}")

    optimizer = torch.optim.Adam([
        {"params": [we.frequencies], "lr": 1e-4},
        {"params": [we.amplitudes, we.harmonics], "lr": 1e-3},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])

    for step in range(50):
        token_ids = torch.randint(0, 100, (16, 16))
        labels = torch.randint(0, 2, (16,))

        optimizer.zero_grad()
        logits = model(token_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    print(f"\nAfter 50 steps:")
    print(f"Freq stats: mean={we.frequencies.data.mean():.4f} std={we.frequencies.data.std():.4f}")
    print(f"Freq range: [{we.frequencies.data.min():.4f}, {we.frequencies.data.max():.4f}]")
    print(f"Amp stats:  mean={we.amplitudes.data.mean():.4f} std={we.amplitudes.data.std():.4f}")
    print(f"Harm stats: mean={we.harmonics.data.mean():.4f} std={we.harmonics.data.std():.4f}")

    # Check: are the freq CHANGES small relative to the initial spread?
    print(f"\nFreq std is large because init spread is large (linspace 0.5 to 64)")
    print(f"The frequencies ARE moving, but the std is dominated by the init spread")

    # Let's look at per-token changes
    torch.manual_seed(42)
    model2 = WaveEmbeddingModel(vocab_size=100, num_classes=2, num_waves=7,
                                 signal_length=256, k_max=8, hidden_dim=64)
    freq_init = model2.wave_embedding.frequencies.data.clone()

    optimizer2 = torch.optim.Adam([
        {"params": [model2.wave_embedding.frequencies], "lr": 1e-4},
        {"params": [model2.wave_embedding.amplitudes, model2.wave_embedding.harmonics], "lr": 1e-3},
        {"params": model2.classifier.parameters(), "lr": 1e-3},
    ])

    for step in range(50):
        token_ids = torch.randint(0, 100, (16, 16))
        labels = torch.randint(0, 2, (16,))
        optimizer2.zero_grad()
        logits = model2(token_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=5.0)
        optimizer2.step()

    freq_diff = (model2.wave_embedding.frequencies.data - freq_init).abs()
    print(f"\nPer-param freq change: mean={freq_diff.mean():.6f} max={freq_diff.max():.6f}")
    print(f"That's {freq_diff.mean() / freq_init.abs().mean() * 100:.4f}% of the mean init value")

    amp_init_val = 1.0  # initialized near 1
    amp_diff = (model2.wave_embedding.amplitudes.data - 1.0).abs()
    print(f"Per-param amp change:  mean={amp_diff.mean():.6f} max={amp_diff.max():.6f}")


if __name__ == "__main__":
    diagnose()
