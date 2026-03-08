"""Phase 1: Train wave embeddings on SST-2 sentiment classification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass

from src.wave_model import WaveEmbeddingModel
from src.tokenizer import build_vocab, Vocab


@dataclass
class TrainConfig:
    num_waves: int = 7
    signal_length: int = 256
    k_max: int = 8
    hidden_dim: int = 128
    max_seq_len: int = 32
    vocab_size: int = 10000
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    freq_lr: float = 1e-3  # same LR now that freqs are normalized [0,1]
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
    model = WaveEmbeddingModel(
        vocab_size=vocab.size,
        num_classes=2,
        num_waves=cfg.num_waves,
        signal_length=cfg.signal_length,
        k_max=cfg.k_max,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    wave_params = sum(p.numel() for p in [
        model.wave_embedding.frequencies,
        model.wave_embedding.amplitudes,
        model.wave_embedding.harmonics,
    ])
    print(f"Total parameters: {total_params:,}")
    print(f"Wave parameters: {wave_params:,} ({wave_params/total_params:.1%})")
    print(f"Params per token: {wave_params // vocab.size} (vs 768 in BERT)")

    # All params at same LR now (freqs are normalized via sigmoid)
    wave_emb = model.wave_embedding
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nTraining for {cfg.num_epochs} epochs...")
    print("-" * 70)

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

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={total_loss/(batch_idx+1):.4f} acc={correct/total:.4f}")

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

        # Log wave param stats — track actual movement
        with torch.no_grad():
            # Actual frequencies after sigmoid scaling
            actual_freqs = torch.sigmoid(wave_emb.frequencies.data) * (cfg.signal_length / 4)
            freq_std = actual_freqs.std().item()
            freq_mean = actual_freqs.mean().item()
            harm_mean = wave_emb.harmonics.data.mean().item()
            harm_std = wave_emb.harmonics.data.std().item()
            amp_std = wave_emb.amplitudes.data.std().item()

        print(f"Epoch {epoch+1}/{cfg.num_epochs} | "
              f"loss={train_loss:.4f} train={train_acc:.4f} val={val_acc:.4f} | "
              f"freq={freq_mean:.1f}+/-{freq_std:.1f} "
              f"harm={harm_mean:.2f}+/-{harm_std:.2f} "
              f"amp_std={amp_std:.3f}")

    print("-" * 70)
    print(f"Final validation accuracy: {val_acc:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, "checkpoints/sst2_wave.pt")
    print("Model saved to checkpoints/sst2_wave.pt")


if __name__ == "__main__":
    main()
