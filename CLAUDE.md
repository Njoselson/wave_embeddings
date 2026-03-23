# Wave Embeddings

## Project Overview

Wave Embeddings is a novel language representation system that replaces high-dimensional dense embeddings with parametric wave representations. Each token is described by a small number of "tone waves," each defined by 3 scalars: frequency, amplitude, and harmonic decay count. Sequences compose via wave interference in Fourier space, enabling dramatically fewer parameters and lower computation than standard transformers.

## Core Hypothesis

Every fundamental communicable concept maps to a fundamental frequency — consistent across languages. Language is a constantly evolving interference pattern of these concept-waves.

## Architecture Summary

### Current: v7 — Spectral State Language Model
- Tokens are spectral PERTURBATIONS to a running state, not static waves
- Each token learns delta_amp + delta_phase at 3 scales × K freqs = 24 params/token (K=4)
- Bidirectional: forward + backward spectral states combined at masked positions
- Scoring via spectral coherence: `state_amp * token_amp * cos(state_phase - token_phase)`
- Efficient cos/sin decomposition: `(B, S*K) @ (S*K, V)` matmul instead of broadcast
- Per-scale learned decay: long-range (0.96), mid-range (0.63), short-range (0.12)
- Training: MLM with cross-entropy loss (single optimizer, no regularizer)
- Files: `src/wave_embedding_v7.py`, `experiments/train_spectral_v7.py`, `experiments/eval_v7.py`

### v6 — Multi-scale Phase-aware Wave Interference (skip-gram)
- 5 params/token (freq_slow, freq_fast, amplitude, phase, scale_mix)
- Skip-gram + negative sampling with wave interference energy scoring
- WS-353 rho: -0.06 (worse than v5 — complexity hurt generalization)

### v5 — Single-frequency Wave LM
- 2 params/token (frequency + amplitude), 7 harmonics
- Language model with cross-entropy, PPL 1,563
- WS-353 rho: 0.23 (best benchmark score before v7)

### Legacy: v3 (complex exponentials), v1 (real sinusoids)

## Experimental Results

### v7 K=4 (24 params/token, 240K total) — WikiText-2 MLM
- PPL: 706 (vs v5's 1,563 — 2.2x better)
- MLM accuracy: 6.8% (random = 0.01%)
- WS-353 rho: -0.05 (hand-picked pairs strong, benchmark weak)
- Strong pairs: war/battle +0.90, city/town +0.71, good/great +0.64, man/woman +0.42
- Problem: function words cluster near 1.0 (the/of +0.99), hurting benchmark correlation
- Decay self-organized: long=0.964, mid=0.633, short=0.122
- Loss plateaued at epoch 5 — 12 effective dims may be the bottleneck

### Key design lessons (v5→v6→v7)
1. Single objective > multiple losses (LM cross-entropy > skip-gram + regularizer)
2. Simplicity wins: fewer optimizer groups, fewer hyperparameters
3. Phase requires non-zero init (zero is a saddle point)
4. cos/sin decomposition enables efficient scoring without (B,V,S,K) broadcast

## Tech Stack

- Python 3.11+ (managed via pyenv, local version set to 3.11.11)
- uv for package/dependency management (use `uv add`, `uv run`)
- PyTorch (primary framework — differentiable FFT support is critical)
- Hugging Face `datasets` for Wikipedia corpus (multilingual)
- pytest for testing (`uv run pytest`)
- NumPy/SciPy for prototyping signal operations

## Key Design Decisions

- v7: tokens as spectral perturbations (24 params/token vs 768+ in transformers)
- MLM with cross-entropy (smoother landscape than skip-gram contrastive)
- Single Adam optimizer, single loss — no regularizers, no param group splits
- cos/sin decomposition for memory-efficient scoring
- WikiText-2 for monolingual experiments; multilingual Wikipedia planned for Phase 3

## Code Conventions

- Use PyTorch conventions (snake_case, nn.Module subclasses)
- Keep model code in `src/` directory
- Keep experiments/notebooks in `experiments/`
- Tests in `tests/`
- Config via dataclasses, not YAML/JSON

## Important Notes

- Phase init must be non-zero (randn * 0.3) — zero init is a saddle point with zero gradient
- cos/sin decomposition: `a*b*cos(p-q) = (a*cos(p))*(b*cos(q)) + (a*sin(p))*(b*sin(q))` — enables matmul scoring
- Spectral coherence similarity is normalized: `coherence / (norm1 * norm2)`
- GPU data loading: move entire dataset + neg table to GPU tensors for 97% utilization
- Vast.ai: RTX 3060 12GB at $0.046/hr sufficient for v7 training; avoid Blackwell GPUs (no PyTorch support)
- Benchmark URLs: WS-353 and SimLex-999 via Dropbox (old GitHub repo is dead)
- SimLex-999 eval returns 0 pairs — format parsing issue unresolved
- See ROADMAP.md for full experimental history and next steps
