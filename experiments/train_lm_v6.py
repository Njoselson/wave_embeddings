"""v6 language model training: multi-scale resonance with decay and gating.

Context = decayed running wave interference of previous tokens.
Next-token logits = resonance with every vocab token.
Phase-aware cross-terms, exponential recency decay, frequency-domain gating.

Two-stage: optionally load contrastive checkpoint → fine-tune with cross-entropy.
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from src.wave_embedding_v6 import WaveLMv6
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
    vocab_size: int = 3000
    min_freq: int = 2
    seq_len: int = 32
    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 1e-3
    freq_lr: float = 3e-4
    eval_every: int = 5
    chunk_size: int = 4
    use_wikitext: bool = True
    device: str = "auto"
    contrastive_checkpoint: str | None = None


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


def compute_perplexity(model, loader, device, chunk_size):
    """Compute perplexity over a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
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
    """Generate text by running wave resonance with decay."""
    model.eval()
    unk = vocab.word2idx["<unk>"]
    tokens = tokenize(prompt_text)
    ids = [vocab.word2idx.get(t, unk) for t in tokens]

    with torch.no_grad():
        for _ in range(max_tokens):
            input_ids = torch.tensor([ids], device=device)
            logits = model(input_ids, chunk_size=len(ids))
            next_logits = logits[0, -1, :]
            # Zero out special tokens
            next_logits[0] = float("-inf")  # <pad>
            next_logits[1] = float("-inf")  # <unk>
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

    return " ".join(vocab.idx2word.get(i, "<?>") for i in ids)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--contrastive-checkpoint", type=str, default=None)
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
    if args.device is not None:
        cfg.device = args.device
    if args.contrastive_checkpoint is not None:
        cfg.contrastive_checkpoint = args.contrastive_checkpoint

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
    )
    log.info("Sequences: %s (len=%d)", f"{len(dataset):,}", cfg.seq_len)

    model = WaveLMv6(
        vocab_size=vocab.size,
        num_harmonics=cfg.num_harmonics,
    ).to(device)

    # Optionally load contrastive checkpoint for embedding weights
    if cfg.contrastive_checkpoint and os.path.exists(cfg.contrastive_checkpoint):
        log.info("Loading contrastive checkpoint: %s", cfg.contrastive_checkpoint)
        ckpt = torch.load(cfg.contrastive_checkpoint, map_location=device, weights_only=False)
        # The contrastive model has embedding.* params — map to our model.embedding.*
        state = ckpt["model_state"]
        emb_state = {}
        for k, v in state.items():
            if k.startswith("embedding."):
                emb_key = k[len("embedding."):]
                if emb_key in dict(model.embedding.named_parameters()):
                    if v.shape == dict(model.embedding.named_parameters())[emb_key].shape:
                        emb_state[emb_key] = v
        if emb_state:
            model.embedding.load_state_dict(emb_state, strict=False)
            log.info("Loaded %d embedding params from contrastive checkpoint", len(emb_state))

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s", f"{total_params:,}")

    # 4 param groups
    optimizer = torch.optim.Adam([
        {"params": [model.embedding.freq_slow], "lr": cfg.freq_lr},
        {"params": [model.embedding.freq_fast], "lr": cfg.freq_lr},
        {"params": [
            model.embedding.amplitudes,
            model.embedding.phase,
            model.embedding.scale_mix,
            model.embedding.decay_slow,
            model.embedding.decay_fast,
        ], "lr": cfg.lr},
        {"params": [
            model.lambda_slow_raw,
            model.lambda_fast_raw,
            model.gate_filter,
            model.gate_bias,
            model.temp,
        ], "lr": cfg.lr},
    ])

    batches_per_epoch = len(loader)
    log.info("Training: %d epochs, %d batches/epoch, %d harmonics, chunk=%d",
             cfg.num_epochs, batches_per_epoch, cfg.num_harmonics, cfg.chunk_size)
    log.info("-" * 80)

    train_start = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch, chunk_size=cfg.chunk_size)

            loss = F.cross_entropy(
                logits[:, 1:, :].reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * batch[:, 1:].numel()
            total_tokens += batch[:, 1:].numel()

            if (batch_idx + 1) % 20 == 0:
                avg = total_loss / total_tokens
                ppl = torch.exp(torch.tensor(avg)).item()
                log.info("  batch %d/%d | loss=%.4f ppl=%.1f",
                         batch_idx + 1, batches_per_epoch, avg, ppl)

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.time() - t0

        with torch.no_grad():
            f_slow_std = model.embedding.freq_slow.data.std().item()
            f_fast_std = model.embedding.freq_fast.data.std().item()
            phase_std = model.embedding.phase.data.std().item()
            mix_mean = torch.sigmoid(model.embedding.scale_mix.data).mean().item()
            lam_slow = model.lambda_slow.item()
            lam_fast = model.lambda_fast.item()

        log.info(
            "Epoch %3d/%d | loss=%.4f ppl=%.1f | f_slow_σ=%.3f f_fast_σ=%.3f | "
            "phase_σ=%.3f mix_μ=%.3f | λ_s=%.4f λ_f=%.4f | %.1fs",
            epoch + 1, cfg.num_epochs, avg_loss, ppl,
            f_slow_std, f_fast_std, phase_std, mix_mean,
            lam_slow, lam_fast, elapsed,
        )

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
    save_path = "checkpoints/wave_lm_v6.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": cfg,
    }, save_path)
    log.info("Saved to %s", save_path)

    # Final generation
    log.info("Final generation:")
    model.eval()
    with torch.no_grad():
        for p in ["the king", "she was", "in the", "it is"]:
            text = generate(model, vocab, device, p, max_tokens=15, temperature=0.8)
            log.info("  %s", text)


if __name__ == "__main__":
    main()
