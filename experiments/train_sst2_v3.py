"""Phase 1 v3: Complex exponential wave embeddings on SST-2."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass

from src.wave_embedding_v3 import WaveModelV3
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
    def __init__(self, texts, labels, vocab, max_len):
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

    print("Loading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2")
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    print("Building vocabulary...")
    vocab = build_vocab(train_texts, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    print(f"Vocabulary size: {vocab.size}")

    train_ds = SentimentDataset(train_texts, train_labels, vocab, cfg.max_seq_len)
    val_ds = SentimentDataset(val_texts, val_labels, vocab, cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = WaveModelV3(
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
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Params per token: {wave_params // vocab.size}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # Snapshot initial frequencies for comparison
    freq_init = model.wave_embedding.frequencies.data.cpu().clone()

    print(f"\nTraining for {cfg.num_epochs} epochs (complex exponential surrogate)...")
    print("-" * 80)

    best_val = 0.0
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for ids, labels in train_loader:
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

        # Track frequency movement
        we = model.wave_embedding
        with torch.no_grad():
            f_data = we.frequencies.data
            f_diff = (f_data.cpu() - freq_init).abs()
            f_mean_change = f_diff.mean().item()
            f_max_change = f_diff.max().item()
            f_std = f_data.std().item()
            f_mean = f_data.mean().item()
            a_std = we.amplitudes.data.std().item()

            # How many tokens moved freq by > 0.5?
            tokens_moved = (f_diff.mean(dim=1) > 0.5).sum().item()

        print(f"Epoch {epoch+1:2d}/{cfg.num_epochs} | "
              f"loss={train_loss:.4f} train={train_acc:.4f} val={val_acc:.4f} | "
              f"freq_change: mean={f_mean_change:.3f} max={f_max_change:.2f} "
              f"moved={tokens_moved}/{vocab.size} | "
              f"f_std={f_std:.2f} a_std={a_std:.3f}")

    print("-" * 80)
    print(f"Best validation accuracy: {best_val:.4f}")

    # Final frequency analysis
    with torch.no_grad():
        f_final = we.frequencies.data.cpu()
        f_diff_final = (f_final - freq_init).abs()
        print(f"\nFrequency movement summary:")
        print(f"  Mean absolute change: {f_diff_final.mean():.4f}")
        print(f"  Max absolute change:  {f_diff_final.max():.4f}")
        print(f"  Tokens with mean change > 0.1: {(f_diff_final.mean(dim=1) > 0.1).sum()}/{vocab.size}")
        print(f"  Tokens with mean change > 0.5: {(f_diff_final.mean(dim=1) > 0.5).sum()}/{vocab.size}")
        print(f"  Tokens with mean change > 1.0: {(f_diff_final.mean(dim=1) > 1.0).sum()}/{vocab.size}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
        "freq_init": freq_init,
    }, "checkpoints/sst2_wave_v3.pt")
    print("Model saved to checkpoints/sst2_wave_v3.pt")


if __name__ == "__main__":
    main()
