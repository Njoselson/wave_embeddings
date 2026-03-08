"""Phase 1 v2: Train frequency-first wave embeddings on SST-2."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass

from src.wave_embedding_v2 import WaveModelV2
from src.tokenizer import build_vocab, Vocab


@dataclass
class TrainConfig:
    num_waves: int = 3
    sample_points: int = 64
    hidden_dim: int = 32
    max_seq_len: int = 32
    vocab_size: int = 10000
    batch_size: int = 64
    num_epochs: int = 15
    lr: float = 1e-3
    min_freq: int = 2


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    cfg = TrainConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SST-2
    print("Loading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    print(f"Vocabulary size: {vocab.size}")

    # Create datasets and loaders
    train_ds = SentimentDataset(train_texts, train_labels, vocab, cfg.max_seq_len)
    val_ds = SentimentDataset(val_texts, val_labels, vocab, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Create model
    model = WaveModelV2(
        vocab_size=vocab.size,
        num_classes=2,
        num_waves=cfg.num_waves,
        sample_points=cfg.sample_points,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    wave_params = sum(p.numel() for p in [
        model.wave_embedding.frequencies,
        model.wave_embedding.amplitudes,
    ])
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Wave parameters: {wave_params:,} ({wave_params/total_params:.1%})")
    print(f"Classifier parameters: {classifier_params:,} ({classifier_params/total_params:.1%})")
    print(f"Params per token: {wave_params // vocab.size} (f + A only)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {cfg.num_epochs} epochs...")
    print(f"Architecture: {cfg.num_waves} waves/token, {cfg.sample_points} sample points, {cfg.hidden_dim} hidden")
    print("-" * 75)

    best_val = 0.0
    for epoch in range(cfg.num_epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (ids, labels) in enumerate(train_loader):
            ids, labels = ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for ids, labels in val_loader:
                ids, labels = ids.to(device), labels.to(device)
                logits = model(ids)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        best_val = max(best_val, val_acc)

        # Track wave param stats
        we = model.wave_embedding
        with torch.no_grad():
            actual_f = torch.nn.functional.softplus(we.frequencies.data)
            f_mean = actual_f.mean().item()
            f_std = actual_f.std().item()
            f_min = actual_f.min().item()
            f_max = actual_f.max().item()
            a_std = we.amplitudes.data.std().item()

            # Check if frequencies are differentiating: compute per-token freq variance
            # High = tokens have different frequencies (good)
            per_token_f_mean = actual_f.mean(dim=1)  # mean freq per token
            token_freq_std = per_token_f_mean.std().item()

        print(f"Epoch {epoch+1:2d}/{cfg.num_epochs} | "
              f"loss={train_loss:.4f} train={train_acc:.4f} val={val_acc:.4f} | "
              f"freq=[{f_min:.1f},{f_max:.1f}] std={f_std:.2f} "
              f"token_spread={token_freq_std:.2f} amp_std={a_std:.3f}")

    print("-" * 75)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Final frequency range: [{f_min:.1f}, {f_max:.1f}], spread across tokens: {token_freq_std:.2f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, "checkpoints/sst2_wave_v2.pt")
    print("Model saved to checkpoints/sst2_wave_v2.pt")


if __name__ == "__main__":
    main()
