"""v5 training: token-level, 1 freq + 1 amp, harmonics for long-range.

Each token = 1 frequency + 1 amplitude = 2 params.
7 harmonics create multi-scale interaction.
Skip-gram training at word level.
"""

import sys
import os
import time
import logging
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.wave_embedding_v5 import (
    SkipGramV5, energy, self_energy, similarity, negative_sampling_loss,
)
from src.skipgram_dataset import SkipGramDataset, tokenize_corpus
from src.tokenizer import build_vocab, tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    num_harmonics: int = 7
    vocab_size: int = 10000
    min_freq: int = 2
    window_size: int = 5
    num_negatives: int = 5
    batch_size: int = 512
    num_epochs: int = 20
    lr: float = 1e-3
    freq_lr: float = 3e-4
    eval_every: int = 5
    use_wikitext: bool = True
    device: str = "cpu"


def load_corpus(use_wikitext: bool):
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))

    if not use_wikitext:
        corpus_path = os.path.join(data_dir, "sample_corpus.txt")
        with open(corpus_path, "r") as f:
            return [l.strip() for l in f if l.strip()]

    wikitext_path = os.path.join(data_dir, "wikitext2_train.txt")
    if os.path.exists(wikitext_path):
        with open(wikitext_path, "r") as f:
            return [l.strip() for l in f if l.strip()]

    log.info("Downloading WikiText-2...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    lines = []
    buf = []
    for text in ds["text"]:
        text = text.strip()
        if not text:
            if buf:
                joined = " ".join(buf)
                if len(joined) > 30:
                    lines.append(joined)
                buf = []
            continue
        if text.startswith("= ") and text.endswith(" ="):
            continue
        buf.append(text)
    if buf:
        joined = " ".join(buf)
        if len(joined) > 30:
            lines.append(joined)

    os.makedirs(data_dir, exist_ok=True)
    with open(wikitext_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    log.info("Cached %d paragraphs", len(lines))
    return lines


def word_pair_energy(w1, w2, model, vocab, device):
    """Energy between two words (as tokens, not characters)."""
    unk = vocab.word2idx["<unk>"]
    id1 = vocab.word2idx.get(w1.lower(), unk)
    id2 = vocab.word2idx.get(w2.lower(), unk)
    if id1 == unk or id2 == unk:
        return None, None
    t1 = torch.tensor([id1], device=device)
    t2 = torch.tensor([id2], device=device)
    f1, A1 = model.embedding.get_harmonics(t1)
    f2, A2 = model.embedding.get_harmonics(t2)
    e = energy(f1, A1, f2, A2).item()
    s = similarity(f1, A1, f2, A2).item()
    return e, s


def log_word_pairs(model, vocab, device, header="Word pairs"):
    pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
        ("the", "a"), ("is", "was"), ("in", "on"),
    ]
    log.info(header)
    model.eval()
    with torch.no_grad():
        for w1, w2 in pairs:
            e, s = word_pair_energy(w1, w2, model, vocab, device)
            if e is not None:
                log.info("  %10s - %-10s: energy=%8.2f  sim=%+.4f", w1, w2, e, s)
            else:
                log.info("  %10s - %-10s: (OOV)", w1, w2)


def demo_running_wave(model, vocab, device):
    """Demo: score next-word candidates given a history."""
    log.info("Running wave demo:")
    model.eval()
    with torch.no_grad():
        sentences = [
            "the king sat on the",
            "the cat chased the",
            "she is very",
        ]
        candidates = ["throne", "dog", "cat", "good", "bad", "the", "table"]
        cand_ids = []
        for w in candidates:
            idx = vocab.word2idx.get(w.lower(), None)
            if idx is not None:
                cand_ids.append(idx)
            else:
                cand_ids.append(vocab.word2idx["<unk>"])
        cand_tensor = torch.tensor(cand_ids, device=device)

        for sent in sentences:
            tokens = tokenize(sent)
            unk = vocab.word2idx["<unk>"]
            hist_ids = [vocab.word2idx.get(t, unk) for t in tokens]
            hist_ids = [i for i in hist_ids if i != unk]
            if not hist_ids:
                continue
            hist_tensor = torch.tensor(hist_ids, device=device)
            scores = model.running_wave_score(hist_tensor, cand_tensor)
            ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1])
            top3 = ", ".join(f"{w}({s:.1f})" for w, s in ranked[:3])
            log.info("  '%s ...' → %s", sent, top3)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.use_wikitext = False
        cfg.device = "cpu"
        cfg.num_epochs = 30
        cfg.batch_size = 128
        cfg.eval_every = 10
        cfg.min_freq = 1
        cfg.vocab_size = 2000
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    if cfg.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info("Device: %s", device)

    t0 = time.time()
    lines = load_corpus(cfg.use_wikitext)
    log.info("Loaded %d paragraphs in %.1fs", len(lines), time.time() - t0)

    t0 = time.time()
    vocab = build_vocab(lines, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    token_ids = tokenize_corpus(lines, vocab)
    total_tokens = sum(len(doc) for doc in token_ids)
    log.info("Vocab: %d tokens, corpus: %s tokens in %.1fs",
             vocab.size, f"{total_tokens:,}", time.time() - t0)

    t0 = time.time()
    dataset = SkipGramDataset(
        token_ids=token_ids,
        vocab=vocab,
        window_size=cfg.window_size,
        num_negatives=cfg.num_negatives,
    )
    log.info("Skip-gram pairs: %s in %.1fs", f"{len(dataset):,}", time.time() - t0)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = SkipGramV5(
        vocab_size=vocab.size,
        num_harmonics=cfg.num_harmonics,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s (%d tokens × 2 + 1 decay)",
             f"{total_params:,}", vocab.size)

    optimizer = torch.optim.Adam([
        {"params": [model.embedding.frequencies], "lr": cfg.freq_lr},
        {"params": [model.embedding.amplitudes], "lr": cfg.lr},
        {"params": [model.embedding.decay], "lr": cfg.lr},
    ])

    freq_init = model.embedding.frequencies.data.cpu().clone()
    batches_per_epoch = len(loader)
    log.info("Training: %d epochs, %d batches/epoch, %d harmonics",
             cfg.num_epochs, batches_per_epoch, cfg.num_harmonics)
    log.info("-" * 80)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        t0 = time.time()

        for target, positive, negatives in loader:
            target = target.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            pos_e, neg_e = model(target, positive, negatives)
            loss = negative_sampling_loss(pos_e, neg_e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                log.info("  batch %d/%d | loss=%.4f",
                         num_batches, batches_per_epoch, total_loss / num_batches)

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        with torch.no_grad():
            f_data = model.embedding.frequencies.data.cpu()
            f_diff = (f_data - freq_init).abs()
            f_mean_change = f_diff.mean().item()
            tokens_moved = (f_diff > 0.1).sum().item()
            decay_val = model.embedding.decay.item()

        log.info("Epoch %3d/%d | loss=%.4f | freq_delta=%.4f moved=%d/%d decay=%.3f | %.1fs",
                 epoch + 1, cfg.num_epochs, avg_loss, f_mean_change,
                 tokens_moved, vocab.size, decay_val, elapsed)

        if (epoch + 1) % cfg.eval_every == 0:
            with torch.no_grad():
                log_word_pairs(model, vocab, device, header="  Word pairs:")
                demo_running_wave(model, vocab, device)
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 80)
    log.info("Done in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/wave_v5.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, save_path)
    log.info("Saved to %s", save_path)

    log_word_pairs(model, vocab, device, header="Final word pairs:")
    demo_running_wave(model, vocab, device)

    # Export — the model IS this table
    csv_path = "checkpoints/wave_v5_params.csv"
    with open(csv_path, "w") as f:
        f.write("token,frequency,amplitude\n")
        with torch.no_grad():
            for word, idx in sorted(vocab.word2idx.items(), key=lambda x: x[1]):
                freq = model.embedding.frequencies[idx].item()
                amp = model.embedding.amplitudes[idx].item()
                f.write(f"{word},{freq:.6f},{amp:.6f}\n")
            f.write(f"# decay={model.embedding.decay.item():.6f}\n")
            f.write(f"# num_harmonics={cfg.num_harmonics}\n")
    log.info("Exported to %s — this IS the model", csv_path)


if __name__ == "__main__":
    main()
