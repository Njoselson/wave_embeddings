"""v4 contrastive training: simplified, all-analytical, no sampling.

Each character = 3 waves × 2 params (f, A) = 6 params/char.
Words = stacked character waves with position frequency shift.
Similarity = analytical sinc formula. No time-domain sampling.
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

from src.wave_embedding_v4 import SkipGramV4, energy, self_energy, similarity
from src.skipgram_dataset import CharSkipGramDataset
from src.tokenizer import build_char_vocab, tokenize_words_to_chars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    num_waves: int = 3
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
    log.info("Cached %d paragraphs", len(lines))
    return lines


def word_energy(w1, w2, model, char_vocab, device):
    """Compute interference energy between two words."""
    unk_id = char_vocab.word2idx["<unk>"]

    def to_ids(w):
        ids = [char_vocab.word2idx.get(c, unk_id) for c in w.lower()]
        return torch.tensor([ids], dtype=torch.long, device=device)

    ids1, ids2 = to_ids(w1), to_ids(w2)
    mask1 = torch.ones(ids1.shape, dtype=torch.bool, device=device)
    mask2 = torch.ones(ids2.shape, dtype=torch.bool, device=device)
    f1, A1 = model.embedding.get_word_params(ids1, mask1)
    f2, A2 = model.embedding.get_word_params(ids2, mask2)
    return energy(f1, A1, f2, A2).item()


def word_sim(w1, w2, model, char_vocab, device):
    """Compute normalized similarity between two words."""
    unk_id = char_vocab.word2idx["<unk>"]

    def to_ids(w):
        ids = [char_vocab.word2idx.get(c, unk_id) for c in w.lower()]
        return torch.tensor([ids], dtype=torch.long, device=device)

    ids1, ids2 = to_ids(w1), to_ids(w2)
    mask1 = torch.ones(ids1.shape, dtype=torch.bool, device=device)
    mask2 = torch.ones(ids2.shape, dtype=torch.bool, device=device)
    f1, A1 = model.embedding.get_word_params(ids1, mask1)
    f2, A2 = model.embedding.get_word_params(ids2, mask2)
    return similarity(f1, A1, f2, A2).item()


def log_word_pairs(model, char_vocab, device, header="Word pairs"):
    pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
        ("cat", "act"), ("dog", "god"),
    ]
    log.info(header)
    model.eval()
    with torch.no_grad():
        for w1, w2 in pairs:
            e = word_energy(w1, w2, model, char_vocab, device)
            s = word_sim(w1, w2, model, char_vocab, device)
            log.info("  %10s - %-10s: energy=%8.2f  sim=%+.4f", w1, w2, e, s)


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
    log.info("Char vocab: %d tokens", char_vocab.size)

    t0 = time.time()
    lines = load_corpus(cfg.use_wikitext)
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

    model = SkipGramV4(
        char_vocab_size=char_vocab.size,
        num_waves=cfg.num_waves,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s (%d chars x %d waves x 2 + 1 position_freq)",
             f"{total_params:,}", char_vocab.size, cfg.num_waves)

    # Separate LR for frequencies (volatile gradients)
    from src.wave_embedding_v4 import negative_sampling_loss
    optimizer = torch.optim.Adam([
        {"params": [model.embedding.frequencies], "lr": cfg.freq_lr},
        {"params": [model.embedding.amplitudes], "lr": cfg.lr},
        {"params": [model.embedding.position_freq], "lr": cfg.lr},
    ])

    freq_init = model.embedding.frequencies.data.cpu().clone()
    batches_per_epoch = len(loader)
    log.info("Training: %d epochs, %d batches/epoch, freq_lr=%s, lr=%s",
             cfg.num_epochs, batches_per_epoch, cfg.freq_lr, cfg.lr)
    log.info("-" * 80)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        t0 = time.time()

        for target_chars, pos_chars, neg_chars, target_mask, pos_mask, neg_mask in loader:
            target_chars = target_chars.to(device)
            pos_chars = pos_chars.to(device)
            neg_chars = neg_chars.to(device)
            target_mask = target_mask.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)

            optimizer.zero_grad()
            pos_e, neg_e = model(
                target_chars, pos_chars, neg_chars,
                target_mask, pos_mask, neg_mask,
            )
            loss = negative_sampling_loss(pos_e, neg_e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                log.info("  batch %d/%d | loss=%.4f",
                         num_batches, batches_per_epoch, total_loss / num_batches)

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        with torch.no_grad():
            f_data = model.embedding.frequencies.data.cpu()
            f_diff = (f_data - freq_init).abs()
            f_mean_change = f_diff.mean().item()
            chars_moved = (f_diff.mean(dim=1) > 0.1).sum().item()
            pos_freq = model.embedding.position_freq.item()

        log.info("Epoch %3d/%d | loss=%.4f | freq_delta=%.4f moved=%2d/%d pos_beta=%.4f | %.1fs",
                 epoch + 1, cfg.num_epochs, avg_loss, f_mean_change,
                 chars_moved, char_vocab.size, pos_freq, elapsed)

        if (epoch + 1) % cfg.eval_every == 0:
            with torch.no_grad():
                log.info("  Char params (f, A):")
                for c in "aeiourstnl":
                    if c in char_vocab.word2idx:
                        idx = char_vocab.word2idx[c]
                        freqs = model.embedding.frequencies[idx].cpu().tolist()
                        amps = model.embedding.amplitudes[idx].cpu().tolist()
                        log.info("    '%s': f=[%+5.2f,%+5.2f,%+5.2f] A=[%.2f,%.2f,%.2f]",
                                 c, freqs[0], freqs[1], freqs[2],
                                 amps[0], amps[1], amps[2])
            log_word_pairs(model, char_vocab, device, header="  Word pairs:")
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 80)
    log.info("Done in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/wave_v4.pt"
    torch.save({
        "model_state": model.state_dict(),
        "char_vocab": char_vocab,
        "config": cfg,
        "freq_init": freq_init,
    }, save_path)
    log.info("Saved to %s", save_path)

    log_word_pairs(model, char_vocab, device, header="Final word pairs:")

    # Export as CSV — this IS the model
    csv_path = "checkpoints/wave_v4_params.csv"
    with open(csv_path, "w") as f:
        K = cfg.num_waves
        header = ",".join(["char"] + [f"f{k}" for k in range(K)] + [f"A{k}" for k in range(K)])
        f.write(header + "\n")
        with torch.no_grad():
            for c, idx in sorted(char_vocab.word2idx.items()):
                freqs = model.embedding.frequencies[idx].cpu().tolist()
                amps = model.embedding.amplitudes[idx].cpu().tolist()
                vals = ",".join([f"{v:.6f}" for v in freqs + amps])
                # Escape comma in char name if needed
                char_repr = repr(c) if c in (",", '"', "\n") else c
                f.write(f"{char_repr},{vals}\n")
        f.write(f"# position_freq={model.embedding.position_freq.item():.6f}\n")
    log.info("Exported params to %s — this IS the model", csv_path)


if __name__ == "__main__":
    main()
