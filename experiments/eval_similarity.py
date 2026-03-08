"""Evaluate trained wave embeddings on word similarity benchmarks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import csv
import io
import urllib.request
from scipy.stats import spearmanr
from dataclasses import dataclass

from src.wave_contrastive import wave_interference_energy, wave_similarity


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/contrastive_wave.pt"
    sample_points: int = 256
    use_normalized: bool = True  # Use wave_similarity vs raw energy


BENCHMARKS = {
    "wordsim353": "https://raw.githubusercontent.com/alexanderpanchenko/sim-eval/master/datasets/wordsim353/wordsim353.tsv",
    "simlex999": "https://raw.githubusercontent.com/alexanderpanchenko/sim-eval/master/datasets/simlex999/simlex999.tsv",
}


def load_benchmark(name: str) -> list[tuple[str, str, float]] | None:
    """Load a word similarity benchmark from URL."""
    url = BENCHMARKS.get(name)
    if url is None:
        print(f"Unknown benchmark: {name}")
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
        print(f"Could not load {name}: {e}")
        return None


def evaluate(checkpoint_path: str, cfg: EvalConfig):
    """Evaluate a checkpoint on word similarity benchmarks."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    # Reconstruct embedding from state dict
    from src.wave_contrastive import SkipGramWaveModel
    model = SkipGramWaveModel(
        vocab_size=vocab.size,
        num_waves=config.num_waves,
        sample_points=cfg.sample_points,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sim_fn = wave_similarity if cfg.use_normalized else wave_interference_energy
    metric_name = "normalized_similarity" if cfg.use_normalized else "raw_energy"
    print(f"Similarity metric: {metric_name}")
    print(f"Vocab size: {vocab.size}")
    print()

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

                id1, id2 = vocab.word2idx[w1], vocab.word2idx[w2]
                f1 = model.target_embedding.frequencies[id1].unsqueeze(0)
                A1 = model.target_embedding.amplitudes[id1].unsqueeze(0)
                f2 = model.target_embedding.frequencies[id2].unsqueeze(0)
                A2 = model.target_embedding.amplitudes[id2].unsqueeze(0)

                s = sim_fn(f1, A1, f2, A2, sample_points=cfg.sample_points)
                human_scores.append(score)
                model_scores.append(s.item())

        covered = len(human_scores)
        total = len(pairs)
        print(f"{bench_name}:")
        print(f"  Coverage: {covered}/{total} pairs ({missing} missing from vocab)")

        if covered >= 10:
            rho, pvalue = spearmanr(human_scores, model_scores)
            print(f"  Spearman rho: {rho:.4f} (p={pvalue:.2e})")
        else:
            print(f"  Too few pairs for evaluation")
        print()

    # Baselines for reference
    print("Reference baselines (approximate):")
    print("  Word2Vec (300d): WordSim-353 rho ~0.65, SimLex-999 rho ~0.44")
    print("  GloVe (300d):    WordSim-353 rho ~0.60, SimLex-999 rho ~0.37")


def main():
    cfg = EvalConfig()
    if len(sys.argv) > 1:
        cfg.checkpoint_path = sys.argv[1]
    evaluate(cfg.checkpoint_path, cfg)


if __name__ == "__main__":
    main()
