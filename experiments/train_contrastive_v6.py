"""v6 contrastive training: skip-gram with multi-scale phase-aware wave interference.

Stage 1 pre-training: learn token embeddings (freq_slow, freq_fast, amplitude, phase, scale_mix)
via skip-gram with negative sampling on WikiText-2.

Adds frequency diversity regularizer to prevent frequency crowding.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.wave_embedding_v6 import (
    SkipGramV6,
    negative_sampling_loss,
    freq_diversity_loss,
    similarity,
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
    freq_diversity_weight: float = 0.01
    eval_every: int = 5
    use_wikitext: bool = True
    device: str = "auto"


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


def word_sim(w1, w2, model, vocab, device):
    """Compute similarity between two words using v6 embeddings."""
    unk = vocab.word2idx.get("<unk>", 1)
    id1 = vocab.word2idx.get(w1, unk)
    id2 = vocab.word2idx.get(w2, unk)
    if id1 == unk or id2 == unk:
        return None

    ids1 = torch.tensor([id1], device=device)
    ids2 = torch.tensor([id2], device=device)
    f1, A1, phi1 = model.embedding.get_harmonics(ids1)
    f2, A2, phi2 = model.embedding.get_harmonics(ids2)
    return similarity(f1, A1, phi1, f2, A2, phi2).item()


def log_word_pairs(model, vocab, device, header="Word pairs"):
    pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
    ]
    log.info(header)
    model.eval()
    with torch.no_grad():
        for w1, w2 in pairs:
            s = word_sim(w1, w2, model, vocab, device)
            if s is not None:
                log.info("  %10s - %-10s: sim=%+.4f", w1, w2, s)
            else:
                log.info("  %10s - %-10s: (OOV)", w1, w2)


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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    log.info("Device: %s", device)

    t0 = time.time()
    lines = load_corpus(cfg.use_wikitext)
    log.info("Loaded %d paragraphs in %.1fs", len(lines), time.time() - t0)

    t0 = time.time()
    vocab = build_vocab(lines, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    log.info("Vocab: %d tokens", vocab.size)

    token_id_seqs = tokenize_corpus(lines, vocab)
    total_tokens = sum(len(seq) for seq in token_id_seqs)
    log.info("Tokenized %s tokens in %.1fs", f"{total_tokens:,}", time.time() - t0)

    t0 = time.time()
    dataset = SkipGramDataset(
        token_ids=token_id_seqs,
        vocab=vocab,
        window_size=cfg.window_size,
        num_negatives=cfg.num_negatives,
    )
    log.info("Dataset: %s pairs in %.1fs", f"{len(dataset):,}", time.time() - t0)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = SkipGramV6(
        vocab_size=vocab.size,
        num_harmonics=cfg.num_harmonics,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s (%d tokens × 5 + 2 decay)", f"{total_params:,}", vocab.size)

    # 4 param groups: freq_slow/fast at lower LR, rest at higher LR
    optimizer = torch.optim.Adam([
        {"params": [model.embedding.freq_slow], "lr": cfg.freq_lr},
        {"params": [model.embedding.freq_fast], "lr": cfg.freq_lr},
        {"params": [
            model.embedding.amplitudes,
            model.embedding.phase,
            model.embedding.scale_mix,
        ], "lr": cfg.lr},
        {"params": [
            model.embedding.decay_slow,
            model.embedding.decay_fast,
        ], "lr": cfg.lr},
    ])

    freq_slow_init = model.embedding.freq_slow.data.cpu().clone()
    freq_fast_init = model.embedding.freq_fast.data.cpu().clone()
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

        for target, pos, neg in loader:
            target = target.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            optimizer.zero_grad()
            pos_e, neg_e = model(target, pos, neg)
            loss = negative_sampling_loss(pos_e, neg_e)

            # Frequency diversity regularizer
            div_loss_slow = freq_diversity_loss(model.embedding.freq_slow)
            div_loss_fast = freq_diversity_loss(model.embedding.freq_fast)
            loss = loss + cfg.freq_diversity_weight * (div_loss_slow + div_loss_fast)

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
            f_slow_data = model.embedding.freq_slow.data.cpu()
            f_fast_data = model.embedding.freq_fast.data.cpu()
            f_slow_delta = (f_slow_data - freq_slow_init).abs().mean().item()
            f_fast_delta = (f_fast_data - freq_fast_init).abs().mean().item()
            f_slow_std = f_slow_data.std().item()
            f_fast_std = f_fast_data.std().item()
            phase_std = model.embedding.phase.data.std().item()
            mix_mean = torch.sigmoid(model.embedding.scale_mix.data).mean().item()

        log.info(
            "Epoch %3d/%d | loss=%.4f | f_slow: Δ=%.4f σ=%.3f | f_fast: Δ=%.4f σ=%.3f | "
            "phase_σ=%.3f mix_μ=%.3f | %.1fs",
            epoch + 1, cfg.num_epochs, avg_loss,
            f_slow_delta, f_slow_std, f_fast_delta, f_fast_std,
            phase_std, mix_mean, elapsed,
        )

        if (epoch + 1) % cfg.eval_every == 0:
            log_word_pairs(model, vocab, device, header="  Word pairs:")
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 80)
    log.info("Done in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/wave_contrastive_v6.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, save_path)
    log.info("Saved to %s", save_path)

    log_word_pairs(model, vocab, device, header="Final word pairs:")


if __name__ == "__main__":
    main()
