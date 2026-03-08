"""Evaluate v6 wave embeddings: similarity benchmarks + ablation + diagnostics.

Usage:
    uv run python experiments/eval_v6.py checkpoints/wave_contrastive_v6.pt
    uv run python experiments/eval_v6.py checkpoints/wave_lm_v6.pt --lm
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import csv
import io
import urllib.request
import logging
from scipy.stats import spearmanr

from src.wave_embedding_v6 import (
    WaveEmbeddingV6,
    SkipGramV6,
    WaveLMv6,
    energy,
    energy_no_phase,
    self_energy,
    similarity,
)

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


def eval_similarity_benchmarks(embedding: WaveEmbeddingV6, vocab, use_phase=True):
    """Evaluate on word similarity benchmarks."""
    log.info("=== Similarity Benchmarks (phase=%s) ===", use_phase)

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
                f1, A1, phi1 = embedding.get_harmonics(ids1)
                f2, A2, phi2 = embedding.get_harmonics(ids2)

                if use_phase:
                    s = similarity(f1, A1, phi1, f2, A2, phi2)
                else:
                    # Phase-ablated: treat as zero phase
                    zero_phi = torch.zeros_like(phi1)
                    s = similarity(f1, A1, zero_phi, f2, A2, zero_phi)

                human_scores.append(score)
                model_scores.append(s.item())

        covered = len(human_scores)
        total = len(pairs)
        log.info("%s: coverage=%d/%d (missing=%d)", bench_name, covered, total, missing)

        if covered >= 10:
            rho, pvalue = spearmanr(human_scores, model_scores)
            log.info("  Spearman rho: %.4f (p=%.2e)", rho, pvalue)
        else:
            log.info("  Too few pairs")

    log.info("Reference: Word2Vec(300d) WS353~0.65, SL999~0.44")


def eval_frequency_diagnostics(embedding: WaveEmbeddingV6, vocab):
    """Log frequency distribution diagnostics."""
    log.info("=== Frequency Diagnostics ===")
    with torch.no_grad():
        f_slow = embedding.freq_slow.data
        f_fast = embedding.freq_fast.data
        phase = embedding.phase.data
        mix = torch.sigmoid(embedding.scale_mix.data)
        amp = embedding.amplitudes.data

        log.info("freq_slow:  mean=%.3f std=%.3f min=%.3f max=%.3f",
                 f_slow.mean(), f_slow.std(), f_slow.min(), f_slow.max())
        log.info("freq_fast:  mean=%.3f std=%.3f min=%.3f max=%.3f",
                 f_fast.mean(), f_fast.std(), f_fast.min(), f_fast.max())
        log.info("phase:      mean=%.3f std=%.3f", phase.mean(), phase.std())
        log.info("scale_mix:  mean=%.3f std=%.3f (sigmoid)", mix.mean(), mix.std())
        log.info("amplitude:  mean=%.3f std=%.3f", amp.mean(), amp.std())
        log.info("decay_slow: %.4f", embedding.decay_slow.item())
        log.info("decay_fast: %.4f", embedding.decay_fast.item())

        # Check for frequency crowding
        f_slow_sorted = f_slow.sort().values
        f_fast_sorted = f_fast.sort().values
        slow_gaps = (f_slow_sorted[1:] - f_slow_sorted[:-1]).abs()
        fast_gaps = (f_fast_sorted[1:] - f_fast_sorted[:-1]).abs()
        log.info("freq_slow gaps: min=%.4f mean=%.4f", slow_gaps.min(), slow_gaps.mean())
        log.info("freq_fast gaps: min=%.4f mean=%.4f", fast_gaps.min(), fast_gaps.mean())


def eval_ablation(embedding: WaveEmbeddingV6, vocab):
    """Ablation: compare similarity with/without phase."""
    log.info("=== Ablation: Phase Impact ===")

    pairs = load_benchmark("wordsim353")
    if pairs is None:
        log.info("Could not load benchmark for ablation")
        return

    scores_with_phase = []
    scores_no_phase = []

    with torch.no_grad():
        for w1, w2, _ in pairs:
            if w1 not in vocab.word2idx or w2 not in vocab.word2idx:
                continue

            ids1 = torch.tensor([vocab.word2idx[w1]])
            ids2 = torch.tensor([vocab.word2idx[w2]])
            f1, A1, phi1 = embedding.get_harmonics(ids1)
            f2, A2, phi2 = embedding.get_harmonics(ids2)

            s_phase = similarity(f1, A1, phi1, f2, A2, phi2).item()
            zero_phi = torch.zeros_like(phi1)
            s_nophase = similarity(f1, A1, zero_phi, f2, A2, zero_phi).item()

            scores_with_phase.append(s_phase)
            scores_no_phase.append(s_nophase)

    if len(scores_with_phase) > 0:
        import numpy as np
        diff = np.array(scores_with_phase) - np.array(scores_no_phase)
        log.info("Phase impact: mean_diff=%.4f std=%.4f max_abs=%.4f",
                 diff.mean(), diff.std(), np.abs(diff).max())
        # If phase has been learned, we expect non-trivial differences
        if np.abs(diff).max() < 1e-6:
            log.info("  → Phase has NOT been learned (all zeros)")
        else:
            log.info("  → Phase IS active (learned non-zero values)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs="?",
                        default="checkpoints/wave_contrastive_v6.pt")
    parser.add_argument("--lm", action="store_true", help="Checkpoint is a WaveLMv6")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    log.info("Loading: %s", args.checkpoint)
    import pickle
    from dataclasses import dataclass

    @dataclass
    class _Config:
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

    num_harmonics = getattr(config, "num_harmonics", 7)

    if args.lm:
        model = WaveLMv6(vocab_size=vocab.size, num_harmonics=num_harmonics)
        model.load_state_dict(ckpt["model_state"])
        embedding = model.embedding
        log.info("Loaded WaveLMv6 (lambda_slow=%.4f, lambda_fast=%.4f)",
                 model.lambda_slow.item(), model.lambda_fast.item())
    else:
        model = SkipGramV6(vocab_size=vocab.size, num_harmonics=num_harmonics)
        model.load_state_dict(ckpt["model_state"])
        embedding = model.embedding

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %s (vocab=%d)", f"{total_params:,}", vocab.size)

    eval_frequency_diagnostics(embedding, vocab)
    print()
    eval_similarity_benchmarks(embedding, vocab, use_phase=True)
    print()
    eval_similarity_benchmarks(embedding, vocab, use_phase=False)
    print()
    eval_ablation(embedding, vocab)


if __name__ == "__main__":
    main()
