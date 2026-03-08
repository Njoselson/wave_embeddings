"""Visualize learned wave embeddings from a trained v6 checkpoint.

Generates 4 plots:
  1. Waveforms: time-domain signals for selected words
  2. Frequency spectra: slow + fast band harmonics as stem plots
  3. Interference heatmap: pairwise similarity matrix
  4. Frequency landscape: all vocab tokens plotted by (freq_slow, freq_fast), colored by amplitude

Usage:
    uv run python experiments/visualize_waves.py checkpoints/wave_contrastive_v6.pt
    uv run python experiments/visualize_waves.py checkpoints/wave_contrastive_v6.pt --words "king queen man woman cat dog"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.wave_embedding_v6 import WaveEmbeddingV6, SkipGramV6, similarity


DEFAULT_WORDS = [
    "king", "queen", "man", "woman",
    "cat", "dog", "good", "bad",
    "sun", "moon", "the", "water",
]


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    cfg = ckpt["config"]
    model = SkipGramV6(vocab_size=vocab.size, num_harmonics=cfg.num_harmonics)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


def get_word_id(word, vocab):
    unk = vocab.word2idx.get("<unk>", 1)
    idx = vocab.word2idx.get(word, unk)
    return idx if idx != unk else None


def synthesize_waveform(freqs, amps, phases, t):
    """Sum sinusoids to produce a time-domain waveform."""
    # freqs, amps, phases: (H,) numpy arrays
    # t: (T,) time axis
    signal = np.zeros_like(t)
    for f, a, phi in zip(freqs, amps, phases):
        signal += a * np.sin(2 * np.pi * f * t + phi)
    return signal


def plot_waveforms(ax, model, vocab, words, t):
    """Plot time-domain waveforms for each word."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(words)))
    for word, color in zip(words, colors):
        wid = get_word_id(word, vocab)
        if wid is None:
            continue
        ids = torch.tensor([wid])
        with torch.no_grad():
            freqs, amps, phases = model.embedding.get_harmonics(ids)
        f_np = freqs[0].numpy()
        a_np = amps[0].numpy()
        p_np = phases[0].numpy()
        signal = synthesize_waveform(f_np, a_np, p_np, t)
        ax.plot(t, signal, label=word, color=color, linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wave Signatures (Time Domain)")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_spectra(ax, model, vocab, words):
    """Plot frequency spectra as stem plots for each word."""
    H = model.embedding.num_harmonics
    n_words = len([w for w in words if get_word_id(w, vocab) is not None])
    colors = plt.cm.tab10(np.linspace(0, 1, len(words)))
    width = 0.8 / max(n_words, 1)

    word_idx = 0
    for word, color in zip(words, colors):
        wid = get_word_id(word, vocab)
        if wid is None:
            continue
        ids = torch.tensor([wid])
        with torch.no_grad():
            freqs, amps, _ = model.embedding.get_harmonics(ids)
        f_np = freqs[0].numpy()
        a_np = np.abs(amps[0].numpy())

        # Slow band (first H) and fast band (last H)
        offset = (word_idx - n_words / 2) * width
        positions = np.arange(2 * H) + offset
        ax.bar(positions[:H], a_np[:H], width=width, color=color,
               alpha=0.7, label=f"{word}")
        ax.bar(positions[H:], a_np[H:], width=width, color=color, alpha=0.4)
        word_idx += 1

    ax.axvline(x=H - 0.5, color="gray", linestyle="--", alpha=0.5, label="slow|fast")
    ax.set_xlabel("Harmonic Index")
    ax.set_ylabel("|Amplitude|")
    ax.set_title("Frequency Spectra (Slow Band | Fast Band)")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_similarity_heatmap(ax, model, vocab, words):
    """Plot pairwise similarity matrix."""
    valid_words = [w for w in words if get_word_id(w, vocab) is not None]
    n = len(valid_words)
    sim_matrix = np.zeros((n, n))

    for i, w1 in enumerate(valid_words):
        for j, w2 in enumerate(valid_words):
            id1 = torch.tensor([get_word_id(w1, vocab)])
            id2 = torch.tensor([get_word_id(w2, vocab)])
            with torch.no_grad():
                f1, A1, phi1 = model.embedding.get_harmonics(id1)
                f2, A2, phi2 = model.embedding.get_harmonics(id2)
                sim = similarity(f1, A1, phi1, f2, A2, phi2).item()
            sim_matrix[i, j] = sim

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(valid_words, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(valid_words, fontsize=8)
    ax.set_title("Wave Interference Similarity")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_frequency_landscape(ax, model, vocab, words):
    """Plot all vocab tokens by (freq_slow, freq_fast), highlight selected words."""
    with torch.no_grad():
        f_slow = model.embedding.freq_slow.numpy()
        f_fast = model.embedding.freq_fast.numpy()
        amps = model.embedding.amplitudes.numpy()

    # Plot all tokens as faint dots
    ax.scatter(f_slow, f_fast, c=np.abs(amps), cmap="viridis",
               alpha=0.15, s=5, edgecolors="none")

    # Highlight selected words
    colors = plt.cm.tab10(np.linspace(0, 1, len(words)))
    for word, color in zip(words, colors):
        wid = get_word_id(word, vocab)
        if wid is None:
            continue
        ax.scatter(f_slow[wid], f_fast[wid], c=[color], s=80,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(word, (f_slow[wid], f_fast[wid]),
                    fontsize=7, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Frequency (Slow Band)")
    ax.set_ylabel("Frequency (Fast Band)")
    ax.set_title("Frequency Landscape (All Vocab)")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--words", type=str, default=None,
                        help="Space-separated words to visualize")
    parser.add_argument("--output", type=str, default="wave_visualization.png",
                        help="Output image path")
    args = parser.parse_args()

    model, vocab = load_checkpoint(args.checkpoint)

    if args.words:
        words = args.words.split()
    else:
        words = [w for w in DEFAULT_WORDS if get_word_id(w, vocab) is not None]

    print(f"Visualizing {len(words)} words: {words}")

    # Time axis for waveform synthesis
    t = np.linspace(0, 2.0, 1000)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Wave Embeddings v6 — Learned Representations", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_waveforms(ax1, model, vocab, words, t)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_spectra(ax2, model, vocab, words)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_similarity_heatmap(ax3, model, vocab, words)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_frequency_landscape(ax4, model, vocab, words)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
