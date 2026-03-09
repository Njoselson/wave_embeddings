"""v7 Spectral State LM: MLM training with spectral perturbations.

Tokens perturb a running spectral state. Bidirectional (forward + backward)
states are combined at masked positions, scored against all vocab via
spectral coherence, trained with cross-entropy loss.

Simple by design: single optimizer, single loss, no regularizer.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from src.wave_embedding_v7 import SpectralStateLM, mask_tokens
from src.tokenizer import build_vocab
from src.skipgram_dataset import tokenize_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    vocab_size: int = 10000
    min_freq: int = 2
    num_scales: int = 3
    K: int = 4
    seq_len: int = 64
    mask_prob: float = 0.15
    batch_size: int = 64
    num_epochs: int = 20
    lr: float = 1e-3
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


def build_mlm_sequences(token_id_seqs, seq_len, pad_id=0):
    """Flatten all token sequences and chunk into fixed-length sequences."""
    # Concatenate all tokens
    all_ids = []
    for seq in token_id_seqs:
        all_ids.extend(seq)

    # Chunk into seq_len pieces
    sequences = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        sequences.append(all_ids[i:i + seq_len])

    return torch.tensor(sequences, dtype=torch.long)  # (N, seq_len)


def word_sim(w1, w2, model, vocab, device):
    """Compute similarity between two words using v7 spectral coherence."""
    unk = vocab.word2idx.get("<unk>", 1)
    id1 = vocab.word2idx.get(w1, unk)
    id2 = vocab.word2idx.get(w2, unk)
    if id1 == unk or id2 == unk:
        return None
    ids1 = torch.tensor([id1], device=device)
    ids2 = torch.tensor([id2], device=device)
    return model.spectral_similarity(ids1, ids2).item()


def log_word_pairs(model, vocab, device, header="Word pairs"):
    pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
        ("war", "battle"), ("city", "town"),
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--num-scales", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-suffix", type=str, default="")
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.use_wikitext = False
        cfg.device = "cpu"
        cfg.num_epochs = 30
        cfg.batch_size = 16
        cfg.eval_every = 10
        cfg.min_freq = 1
        cfg.vocab_size = 2000
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.num_scales is not None:
        cfg.num_scales = args.num_scales
    if args.K is not None:
        cfg.K = args.K
    if args.lr is not None:
        cfg.lr = args.lr

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

    # Load and tokenize corpus
    t0 = time.time()
    lines = load_corpus(cfg.use_wikitext)
    log.info("Loaded %d paragraphs in %.1fs", len(lines), time.time() - t0)

    t0 = time.time()
    vocab = build_vocab(lines, max_size=cfg.vocab_size, min_freq=cfg.min_freq)
    log.info("Vocab: %d tokens", vocab.size)

    token_id_seqs = tokenize_corpus(lines, vocab)
    total_tokens = sum(len(seq) for seq in token_id_seqs)
    log.info("Tokenized %s tokens in %.1fs", f"{total_tokens:,}", time.time() - t0)

    # Build fixed-length sequences for MLM
    t0 = time.time()
    all_sequences = build_mlm_sequences(token_id_seqs, cfg.seq_len)
    log.info("Built %s sequences of length %d in %.1fs",
             f"{all_sequences.shape[0]:,}", cfg.seq_len, time.time() - t0)

    # Move all data to device
    all_sequences = all_sequences.to(device)
    num_seqs = all_sequences.shape[0]
    log.info("Loaded %s sequences onto %s", f"{num_seqs:,}", device)

    # Build model
    model = SpectralStateLM(
        vocab_size=vocab.size,
        num_scales=cfg.num_scales,
        K=cfg.K,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    params_per_tok = model.embedding.params_per_token
    log.info("Parameters: %s (%d tokens × %d + %d decay + 2 global)",
             f"{total_params:,}", vocab.size, params_per_tok,
             cfg.num_scales)

    # Single optimizer — simplicity wins
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    batches_per_epoch = (num_seqs + cfg.batch_size - 1) // cfg.batch_size
    log.info("Training: %d epochs, %d batches/epoch, lr=%s, mask_prob=%.2f",
             cfg.num_epochs, batches_per_epoch, cfg.lr, cfg.mask_prob)
    log.info("-" * 80)

    pad_id = vocab.word2idx.get("<pad>", 0)
    unk_id = vocab.word2idx.get("<unk>", 1)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_masked = 0
        num_batches = 0
        t0 = time.time()

        # Shuffle sequences each epoch
        perm = torch.randperm(num_seqs, device=device)

        for start in range(0, num_seqs - cfg.batch_size + 1, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            token_ids = all_sequences[batch_idx]  # (B, seq_len)

            # Create mask
            mask = mask_tokens(token_ids, mask_prob=cfg.mask_prob,
                               pad_id=pad_id, unk_id=unk_id)

            # Skip if nothing masked
            num_masked = mask.sum().item()
            if num_masked == 0:
                continue

            optimizer.zero_grad()

            # Forward: get logits at masked positions
            logits = model(token_ids, mask)  # (num_masked, V)
            targets = model.get_targets(token_ids, mask)  # (num_masked,)

            loss = F.cross_entropy(logits, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * num_masked
            total_masked += num_masked

            # Track accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).sum().item()

            num_batches += 1

            if num_batches % 200 == 0:
                avg = total_loss / max(total_masked, 1)
                acc = total_correct / max(total_masked, 1) * 100
                log.info("  batch %d/%d | loss=%.4f | acc=%.1f%%",
                         num_batches, batches_per_epoch, avg, acc)

        avg_loss = total_loss / max(total_masked, 1)
        ppl = np.exp(min(avg_loss, 20))  # cap to avoid overflow
        accuracy = total_correct / max(total_masked, 1) * 100
        elapsed = time.time() - t0

        with torch.no_grad():
            decay = model.embedding.decay.cpu()
            decay_str = " ".join(f"{d:.3f}" for d in decay)

        log.info(
            "Epoch %3d/%d | loss=%.4f | PPL=%.0f | acc=%.1f%% | "
            "decay=[%s] | temp=%.3f | %.1fs",
            epoch + 1, cfg.num_epochs, avg_loss, ppl, accuracy,
            decay_str, model.temperature.abs().item(), elapsed,
        )

        if (epoch + 1) % cfg.eval_every == 0:
            log_word_pairs(model, vocab, device, header="  Word pairs:")
            model.train()

    total_time = time.time() - train_start
    log.info("-" * 80)
    log.info("Done in %.1fs (%.1fs/epoch)", total_time, total_time / cfg.num_epochs)

    os.makedirs("checkpoints", exist_ok=True)
    suffix = args.save_suffix if args.save_suffix else ""
    save_path = f"checkpoints/wave_spectral_v7{suffix}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, save_path)
    log.info("Saved to %s", save_path)

    log_word_pairs(model, vocab, device, header="Final word pairs:")


if __name__ == "__main__":
    main()
