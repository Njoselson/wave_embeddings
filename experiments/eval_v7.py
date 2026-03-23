"""Evaluate v7 spectral state embeddings on word similarity benchmarks.

Usage:
    python experiments/eval_v7.py checkpoints/wave_spectral_v7_K4.pt
    python experiments/eval_v7.py checkpoints/wave_spectral_v7_K8.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import csv
import io
import urllib.request
import logging
import pickle
from dataclasses import dataclass
from scipy.stats import spearmanr

from src.wave_embedding_v7 import SpectralStateLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


BENCHMARKS = {
    "wordsim353": "https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1",
    "simlex999": "https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1",
}


def load_benchmark(name: str):
    url = BENCHMARKS.get(name)
    if url is None:
        return None
    try:
        response = urllib.request.urlopen(url, timeout=15)
        content = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(content), delimiter="\t")
        pairs = []
        for row in reader:
            if len(row) >= 3:
                try:
                    w1, w2, score = row[0].lower(), row[1].lower(), float(row[2])
                    pairs.append((w1, w2, score))
                except ValueError:
                    continue
        return pairs
    except Exception as e:
        log.warning("Could not load %s: %s", name, e)
        return None


def eval_similarity_benchmarks(model: SpectralStateLM, vocab):
    """Evaluate on word similarity benchmarks using spectral_similarity."""
    log.info("=== Similarity Benchmarks ===")

    for bench_name in BENCHMARKS:
        pairs = load_benchmark(bench_name)
        if pairs is None:
            continue

        human_scores = []
        model_scores = []
        missing = 0

        with torch.no_grad():
            for w1, w2, score in pairs:
                if w1 not in vocab.word2idx or w2 not in vocab.word2idx:
                    missing += 1
                    continue

                ids1 = torch.tensor([vocab.word2idx[w1]])
                ids2 = torch.tensor([vocab.word2idx[w2]])
                s = model.spectral_similarity(ids1, ids2).item()

                human_scores.append(score)
                model_scores.append(s)

        covered = len(human_scores)
        total = len(pairs)
        log.info("%s: coverage=%d/%d (missing=%d)", bench_name, covered, total, missing)

        if covered >= 10:
            rho, pvalue = spearmanr(human_scores, model_scores)
            log.info("  Spearman rho: %.4f (p=%.4f)", rho, pvalue)
        else:
            log.info("  Too few pairs")

    log.info("Reference: Word2Vec(300d) WS353~0.65, SL999~0.44")
    log.info("Reference: WaveLM v5 (2 params/tok) WS353=0.23")
    log.info("Reference: Wave v6 (5 params/tok) WS353=-0.06")


def eval_diagnostics(model: SpectralStateLM):
    """Log embedding parameter diagnostics."""
    log.info("=== Embedding Diagnostics ===")
    emb = model.embedding
    with torch.no_grad():
        d_amp = emb.delta_amp.data
        d_phase = emb.delta_phase.data
        decay = emb.decay

        log.info("delta_amp:   mean=%.4f std=%.4f min=%.4f max=%.4f",
                 d_amp.mean(), d_amp.std(), d_amp.min(), d_amp.max())
        log.info("delta_phase: mean=%.4f std=%.4f min=%.4f max=%.4f",
                 d_phase.mean(), d_phase.std(), d_phase.min(), d_phase.max())
        for s in range(emb.num_scales):
            log.info("  scale %d decay=%.4f | amp_std=%.4f phase_std=%.4f",
                     s, decay[s].item(),
                     d_amp[:, s, :].std().item(),
                     d_phase[:, s, :].std().item())
        log.info("temperature: %.4f", model.temperature.abs().item())
        log.info("combine_weight: %.4f (sigmoid=%.4f)",
                 model.combine_weight.item(),
                 torch.sigmoid(model.combine_weight).item())


def eval_word_pairs(model: SpectralStateLM, vocab):
    """Log hand-picked word pair similarities."""
    log.info("=== Word Pair Similarities ===")
    pairs = [
        ("king", "queen"), ("king", "table"), ("good", "great"),
        ("good", "bad"), ("cat", "dog"), ("cat", "the"),
        ("man", "woman"), ("boy", "girl"), ("sun", "moon"),
        ("war", "battle"), ("city", "town"), ("he", "she"),
        ("is", "was"), ("the", "of"), ("baby", "mother"),
    ]
    unk = vocab.word2idx.get("<unk>", 1)
    with torch.no_grad():
        for w1, w2 in pairs:
            id1 = vocab.word2idx.get(w1, unk)
            id2 = vocab.word2idx.get(w2, unk)
            if id1 == unk or id2 == unk:
                log.info("  %10s - %-10s: (OOV)", w1, w2)
                continue
            ids1 = torch.tensor([id1])
            ids2 = torch.tensor([id2])
            s = model.spectral_similarity(ids1, ids2).item()
            log.info("  %10s - %-10s: sim=%+.4f", w1, w2, s)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?",
                        default="checkpoints/wave_spectral_v7_K4.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    log.info("Loading: %s", args.checkpoint)

    @dataclass
    class _Config:
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

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "Config" and module == "__main__":
                return _Config
            return super().find_class(module, name)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False,
                      pickle_module=type("M", (), {
                          "Unpickler": _Unpickler,
                          "load": lambda f: _Unpickler(f).load(),
                      }))
    vocab = ckpt["vocab"]
    config = ckpt["config"]

    num_scales = getattr(config, "num_scales", 3)
    K = getattr(config, "K", 4)

    model = SpectralStateLM(vocab_size=vocab.size, num_scales=num_scales, K=K)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s (vocab=%d, scales=%d, K=%d, %d params/token)",
             f"{total_params:,}", vocab.size, num_scales, K,
             model.embedding.params_per_token)

    eval_diagnostics(model)
    print()
    eval_word_pairs(model, vocab)
    print()
    eval_similarity_benchmarks(model, vocab)


if __name__ == "__main__":
    main()
