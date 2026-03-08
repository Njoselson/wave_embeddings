"""v5 language model training: running wave predicts next token.

Context = accumulated wave interference of all previous tokens.
Next-token logits = resonance (cross-term) with every vocab token.
Full softmax, no negative sampling. Cost ≈ one transformer layer.
"""

import sys
import os
import time
import logging
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from src.wave_embedding_v5 import WaveLM
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
    seq_len: int = 64
    batch_size: int = 32
    num_epochs: int = 20
    lr: float = 1e-3
    freq_lr: float = 3e-4
    eval_every: int = 5
    chunk_size: int = 16
    use_wikitext: bool = True
    device: str = "auto"


def auto_device() -> torch.device:
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SequenceDataset(Dataset):
    """Chop tokenized corpus into fixed-length sequences."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.num_sequences = max(0, len(self.data) - seq_len) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start : start + self.seq_len]


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


def compute_perplexity(model, loader, device, chunk_size, use_amp):
    """Compute perplexity over a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch, chunk_size=chunk_size)
            loss = F.cross_entropy(
                logits[:, 1:, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += batch[:, 1:].numel()
    avg_loss = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_loss)).item()


def generate(model, vocab, device, prompt_text, max_tokens=20, temperature=1.0):
    """Generate text by running wave resonance."""
    model.eval()
    unk = vocab.word2idx["<unk>"]
    tokens = tokenize(prompt_text)
    ids = [vocab.word2idx.get(t, unk) for t in tokens]

    with torch.no_grad():
        for _ in range(max_tokens):
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids, chunk_size=len(ids))
            next_logits = logits[0, -1, :] / temperature
            next_logits[0] = float("-inf")  # <pad>
            next_logits[1] = float("-inf")  # <unk>
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

    return " ".join(vocab.idx2word.get(i, "<?>") for i in ids)


def export_csv(model, vocab, path):
    """Export trained (frequency, amplitude) pairs as CSV lookup table."""
    freqs = model.embedding.frequencies.detach().cpu()
    amps = model.embedding.amplitudes.detach().cpu()
    decay = model.embedding.decay.detach().cpu().item()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "frequency", "amplitude"])
        for i in range(model.embedding.vocab_size):
            token = vocab.idx2word.get(i, f"<id_{i}>")
            writer.writerow([i, token, f"{freqs[i].item():.6f}", f"{amps[i].item():.6f}"])

    log.info("Exported CSV: %s (%d tokens, decay=%.4f)", path, model.embedding.vocab_size, decay)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.use_wikitext = False
        cfg.device = "cpu"
        cfg.num_epochs = 20
        cfg.batch_size = 16
        cfg.seq_len = 32
        cfg.eval_every = 5
        cfg.min_freq = 1
        cfg.vocab_size = 2000
        cfg.chunk_size = 8
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.device is not None:
        cfg.device = args.device

    # Resolve device
    if cfg.device == "auto":
        device = auto_device()
    else:
        device = torch.device(cfg.device)
    log.info("Device: %s", device)

    # GPU-optimized defaults
    if device.type == "cuda" and not args.quick:
        if args.batch_size is None:
            cfg.batch_size = 32
        if args.chunk_size is None:
            cfg.chunk_size = 4

    # Mixed precision: enabled on CUDA by default, disabled elsewhere
    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp:
        log.info("Mixed precision (float16) enabled")

    t0 = time.time()
    lines = load_corpus(cfg.use_wikitext)
    log.info("Loaded %d paragraphs in %.1fs", len(lines), time.time() - t0)

    t0 = time.time()
    vocab = build_vocab(lines, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    log.info("Vocab: %d tokens", vocab.size)

    all_ids = []
    for line in lines:
        tokens = tokenize(line)
        ids = [vocab.word2idx.get(t, vocab.word2idx["<unk>"]) for t in tokens]
        all_ids.extend(ids)
    log.info("Corpus: %s tokens in %.1fs", f"{len(all_ids):,}", time.time() - t0)

    dataset = SequenceDataset(all_ids, cfg.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Sequences: %s (len=%d)", f"{len(dataset):,}", cfg.seq_len)

    model = WaveLM(
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

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    batches_per_epoch = len(loader)
    log.info("Training: %d epochs, %d batches/epoch, batch_size=%d, %d harmonics, chunk=%d",
             cfg.num_epochs, batches_per_epoch, cfg.batch_size, cfg.num_harmonics, cfg.chunk_size)
    log.info("-" * 80)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch, chunk_size=cfg.chunk_size)
                loss = F.cross_entropy(
                    logits[:, 1:, :].reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch[:, 1:].numel()
            total_tokens += batch[:, 1:].numel()

            if (batch_idx + 1) % 20 == 0:
                avg = total_loss / total_tokens
                ppl = torch.exp(torch.tensor(avg)).item()
                # ETA
                elapsed = time.time() - train_start
                batches_done = epoch * batches_per_epoch + batch_idx + 1
                batches_total = cfg.num_epochs * batches_per_epoch
                if batches_done > 0:
                    eta_s = elapsed / batches_done * (batches_total - batches_done)
                    eta_str = f"{eta_s/60:.0f}m" if eta_s > 60 else f"{eta_s:.0f}s"
                else:
                    eta_str = "?"
                log.info("  batch %d/%d | loss=%.4f ppl=%.1f | ETA %s",
                         batch_idx + 1, batches_per_epoch, avg, ppl, eta_str)

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.time() - t0

        with torch.no_grad():
            decay_val = model.embedding.decay.item()
            f_std = model.embedding.frequencies.data.std().item()

        log.info("Epoch %3d/%d | loss=%.4f ppl=%.1f | f_std=%.3f decay=%.3f | %.1fs",
                 epoch + 1, cfg.num_epochs, avg_loss, ppl, f_std, decay_val, elapsed)

        if (epoch + 1) % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                prompts = ["the king", "she was", "in the"]
                for p in prompts:
                    text = generate(model, vocab, device, p, max_tokens=10, temperature=0.8)
                    log.info("  '%s' → %s", p, text)
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 80)
    log.info("Done in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/wave_lm_v5.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, save_path)
    log.info("Saved checkpoint to %s", save_path)

    # Export CSV lookup table
    csv_path = "checkpoints/wave_v5_params.csv"
    export_csv(model, vocab, csv_path)

    # Final generation
    log.info("Final generation:")
    model.eval()
    with torch.no_grad():
        for p in ["the king", "she was", "in the", "it is"]:
            text = generate(model, vocab, device, p, max_tokens=15, temperature=0.8)
            log.info("  %s", text)


if __name__ == "__main__":
    main()
