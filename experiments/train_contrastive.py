"""Char-level wave contrastive training with harmonics + position encoding.

Each character has 3 waves × (frequency, amplitude, harmonic_decay) = 9 params/char.
Words are composed by summing character signals (fundamentals + harmonics)
with position-dependent phase shifts, so "cat" ≠ "act".
Skip-gram at word level, gradients flow back to character params.
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

from src.wave_contrastive import (
    HarmonicCharSkipGramModel,
    compose_word_signal,
    word_energy_discrete,
    negative_sampling_loss,
)
from src.skipgram_dataset import CharSkipGramDataset
from src.tokenizer import build_char_vocab, tokenize_words_to_chars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class CharContrastiveConfig:
    num_waves: int = 3
    num_harmonics: int = 4
    sample_points: int = 64
    window_size: int = 5
    num_negatives: int = 5
    batch_size: int = 512
    num_epochs: int = 20
    lr: float = 1e-3
    freq_lr: float = 3e-4
    decay_lr: float = 1e-3
    eval_every: int = 5
    use_wikitext: bool = True  # False = use small sample corpus
    device: str = "cpu"  # "cpu", "mps", or "auto"


def load_corpus():
    """Load training corpus. Downloads WikiText-2 if no local cache exists."""
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    wikitext_path = os.path.join(data_dir, "wikitext2_train.txt")

    if os.path.exists(wikitext_path):
        with open(wikitext_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines

    log.info("Downloading WikiText-2 (first run only)...")
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
    log.info("Cached %d paragraphs (%d KB)", len(lines), os.path.getsize(wikitext_path) // 1024)

    return lines


def compute_word_pair_energy(w1, w2, model, char_vocab, device):
    """Compute interference energy between two words."""
    unk_id = char_vocab.word2idx["<unk>"]

    def to_ids(w):
        ids = [char_vocab.word2idx.get(c, unk_id) for c in w.lower()]
        return torch.tensor([ids], dtype=torch.long, device=device)

    ids1, ids2 = to_ids(w1), to_ids(w2)
    mask1 = torch.ones(ids1.shape, dtype=torch.bool, device=device)
    mask2 = torch.ones(ids2.shape, dtype=torch.bool, device=device)
    r1, i1 = model.compose(ids1, mask1)
    r2, i2 = model.compose(ids2, mask2)
    return word_energy_discrete(r1, i1, r2, i2).item()


def log_word_pairs(model, char_vocab, device, header="Word pair energies"):
    sample_pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
        ("cat", "act"), ("dog", "god"),  # anagram test — should differ now!
    ]
    log.info(header)
    model.eval()
    with torch.no_grad():
        for w1, w2 in sample_pairs:
            e = compute_word_pair_energy(w1, w2, model, char_vocab, device)
            e1 = compute_word_pair_energy(w1, w1, model, char_vocab, device)
            e2 = compute_word_pair_energy(w2, w2, model, char_vocab, device)
            norm = 2 * math.sqrt(e1 * e2) if e1 > 0 and e2 > 0 else 1.0
            sim = (e - e1 - e2) / norm if norm > 0 else 0
            log.info("  %10s - %-10s: energy=%8.2f  sim=%+.3f", w1, w2, e, sim)


def load_sample_corpus():
    """Load small local corpus for quick experiments."""
    corpus_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_corpus.txt"))
    with open(corpus_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use small sample corpus on CPU")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = CharContrastiveConfig()

    if args.quick:
        cfg.use_wikitext = False
        cfg.device = "cpu"
        cfg.num_epochs = 30
        cfg.batch_size = 128
        cfg.eval_every = 10

    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    if cfg.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    log.info("Device: %s", device)

    char_vocab = build_char_vocab()
    log.info("Character vocab size: %d", char_vocab.size)

    t0 = time.time()
    if cfg.use_wikitext:
        lines = load_corpus()
    else:
        lines = load_sample_corpus()
    log.info("Loaded %d paragraphs in %.1fs", len(lines), time.time() - t0)

    t0 = time.time()
    word_sequences = [tokenize_words_to_chars(line, char_vocab) for line in lines]
    total_words = sum(len(seq) for seq in word_sequences)
    log.info("Tokenized %s words in %.1fs", f"{total_words:,}", time.time() - t0)

    t0 = time.time()
    dataset = CharSkipGramDataset(
        word_sequences=word_sequences,
        window_size=cfg.window_size,
        num_negatives=cfg.num_negatives,
    )
    log.info("Dataset: %s positions, %s unique words in %.1fs",
             f"{len(dataset):,}", f"{len(dataset.all_words):,}", time.time() - t0)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )

    model = HarmonicCharSkipGramModel(
        char_vocab_size=char_vocab.size,
        num_waves=cfg.num_waves,
        num_harmonics=cfg.num_harmonics,
        sample_points=cfg.sample_points,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Total parameters: %s", f"{total_params:,}")
    log.info("  %d chars × %d waves × 3 params (f, A, d) + 1 position_freq",
             char_vocab.size, cfg.num_waves)
    log.info("  %d harmonics per wave, %d sample points", cfg.num_harmonics, cfg.sample_points)

    # Separate LR for frequencies, amplitudes, decays, position
    optimizer = torch.optim.Adam([
        {"params": [model.embedding.frequencies], "lr": cfg.freq_lr},
        {"params": [model.embedding.amplitudes], "lr": cfg.lr},
        {"params": [model.embedding.decays], "lr": cfg.decay_lr},
        {"params": [model.embedding.position_freq], "lr": cfg.lr},
    ])

    freq_init = model.embedding.frequencies.data.cpu().clone()

    batches_per_epoch = len(loader)
    log.info("Training for %d epochs (%d batches/epoch)", cfg.num_epochs, batches_per_epoch)
    log.info("freq_lr=%s, amp_lr=%s, decay_lr=%s, num_neg=%d",
             cfg.freq_lr, cfg.lr, cfg.decay_lr, cfg.num_negatives)
    log.info("-" * 90)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        t0 = time.time()

        t_data = 0
        t_forward = 0
        t_backward = 0
        t_collate = time.time()

        for target_chars, pos_chars, neg_chars, target_mask, pos_mask, neg_mask in loader:
            t_data += time.time() - t_collate

            target_chars = target_chars.to(device)
            pos_chars = pos_chars.to(device)
            neg_chars = neg_chars.to(device)
            target_mask = target_mask.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)

            optimizer.zero_grad()
            t1 = time.time()
            pos_energy, neg_energy = model(
                target_chars, pos_chars, neg_chars,
                target_mask, pos_mask, neg_mask,
            )
            loss = negative_sampling_loss(pos_energy, neg_energy)
            t_forward += time.time() - t1

            t1 = time.time()
            loss.backward()
            t_backward += time.time() - t1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                log.info("  batch %d/%d | loss=%.4f | data=%.1fs fwd=%.1fs bwd=%.1fs",
                         num_batches, batches_per_epoch, total_loss / num_batches,
                         t_data, t_forward, t_backward)

            t_collate = time.time()

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        with torch.no_grad():
            f_data = model.embedding.frequencies.data.cpu()
            f_diff = (f_data - freq_init).abs()
            f_mean_change = f_diff.mean().item()
            f_std = f_data.std().item()
            chars_moved = (f_diff.mean(dim=1) > 0.1).sum().item()
            d_mean = model.embedding.decays.data.cpu().mean().item()
            pos_freq = model.embedding.position_freq.item()

        log.info("Epoch %3d/%d | loss=%.4f | freq_Δ=%.4f moved=%2d/%d f_std=%.3f "
                 "d_mean=%.3f pos_β=%.4f | %.1fs (data=%.1f fwd=%.1f bwd=%.1f)",
                 epoch + 1, cfg.num_epochs, avg_loss, f_mean_change,
                 chars_moved, char_vocab.size, f_std, d_mean, pos_freq,
                 elapsed, t_data, t_forward, t_backward)

        if (epoch + 1) % cfg.eval_every == 0:
            with torch.no_grad():
                log.info("  Char params (f, A, d):")
                for c in "aeiourstnl":
                    if c in char_vocab.word2idx:
                        idx = char_vocab.word2idx[c]
                        freqs = model.embedding.frequencies[idx].cpu().tolist()
                        amps = model.embedding.amplitudes[idx].cpu().tolist()
                        decs = model.embedding.decays[idx].cpu().tolist()
                        log.info("    '%s': f=[%+5.2f,%+5.2f,%+5.2f] "
                                 "A=[%.2f,%.2f,%.2f] d=[%.2f,%.2f,%.2f]",
                                 c, freqs[0], freqs[1], freqs[2],
                                 amps[0], amps[1], amps[2],
                                 decs[0], decs[1], decs[2])
            log_word_pairs(model, char_vocab, device, header="  Word pair energies:")
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 90)
    log.info("Training completed in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/char_contrastive_wave.pt"
    torch.save({
        "model_state": model.state_dict(),
        "char_vocab": char_vocab,
        "config": cfg,
        "freq_init": freq_init,
    }, save_path)
    log.info("Model saved to %s", save_path)

    log_word_pairs(model, char_vocab, device, header="Final word pair energies:")

    log.info("Self-energy (same word):")
    model.eval()
    with torch.no_grad():
        for w in ["cat", "act", "dog", "god", "king", "queen", "the", "good", "great"]:
            e = compute_word_pair_energy(w, w, model, char_vocab, device)
            log.info("  %10s: energy=%.4f", w, e)

    log.info("Character frequency clusters (sorted by mean freq):")
    with torch.no_grad():
        f_data = model.embedding.frequencies.data.cpu()
        d_data = model.embedding.decays.data.cpu()
        char_freqs = []
        for c in "abcdefghijklmnopqrstuvwxyz":
            if c in char_vocab.word2idx:
                idx = char_vocab.word2idx[c]
                mean_f = f_data[idx].mean().item()
                mean_d = d_data[idx].mean().item()
                char_freqs.append((c, mean_f, mean_d, f_data[idx].tolist()))
        char_freqs.sort(key=lambda x: x[1])
        for c, mf, md, fs in char_freqs:
            log.info("  '%s': mean_f=%+.3f d=%.2f [%+.2f,%+.2f,%+.2f]",
                     c, mf, md, fs[0], fs[1], fs[2])


if __name__ == "__main__":
    main()
