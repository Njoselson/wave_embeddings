# Wave Embeddings

## Project Overview

Wave Embeddings is a novel language representation system that replaces high-dimensional dense embeddings with parametric wave representations. Each token is described by a small number of "tone waves," each defined by 3 scalars: frequency, amplitude, and harmonic decay count. Sequences compose via wave interference in Fourier space, enabling dramatically fewer parameters and lower computation than standard transformers.

## Core Hypothesis

Every fundamental communicable concept maps to a fundamental frequency — consistent across languages. Language is a constantly evolving interference pattern of these concept-waves.

## Architecture Summary

### Current: v3 — Complex Exponential Wave Embeddings
- Each token maps to 3 complex exponential waves (tunable via `num_waves`)
- Each wave = 2 learnable scalars: frequency (f), amplitude (A) — 6 params/token
- Uses complex exponentials z(t) = A * exp(j * 2pi * f * t) instead of real sinusoids
- Key finding: frequencies only learn via complex exponentials (Hayes et al. 2022), not real sinusoids
- Sequence composition: additive interference of per-token signals
- Training: backprop through differentiable complex exponentials
- Similarity metric: wave interference energy — constructive interference when frequencies match
- Contrastive training: skip-gram with negative sampling using interference energy as score

### Legacy: v1 (real sinusoids + harmonics)
- 7 tone waves × 3 params (f, A, H) = 21 params/token
- Harmonics with sigmoid envelope for differentiability
- Frequencies did not learn meaningfully due to non-convex sin() landscape

## Tech Stack

- Python 3.11+ (managed via pyenv, local version set to 3.11.11)
- uv for package/dependency management (use `uv add`, `uv run`)
- PyTorch (primary framework — differentiable FFT support is critical)
- Hugging Face `datasets` for Wikipedia corpus (multilingual)
- pytest for testing (`uv run pytest`)
- NumPy/SciPy for prototyping signal operations

## Key Design Decisions

- v3: 3 complex exponential waves per token (6 params vs 768+ in transformers)
- Complex exponentials provide smooth loss landscape for frequency learning
- Frequency initialization: randn * 3.0 for diverse spread
- Wave interference energy as similarity metric (physically motivated)
- Contrastive (skip-gram) training to force frequency differentiation across tokens
- Multilingual Wikipedia as primary dataset to test universal-frequency hypothesis

## Code Conventions

- Use PyTorch conventions (snake_case, nn.Module subclasses)
- Keep model code in `src/` directory
- Keep experiments/notebooks in `experiments/`
- Tests in `tests/`
- Config via dataclasses, not YAML/JSON

## Important Notes

- `torch.fft.rfft` and `torch.fft.irfft` are fully differentiable — gradients flow through them
- Complex exponentials exp(j*w*t) provide convex loss landscape for frequency (Hayes et al. 2022)
- Frequency parameter gradients can be volatile — use lower learning rate or separate param groups for frequencies
- Wave interference energy: E = A1² + A2² + 2·A1·A2·sinc(Δf) — cross-term is the similarity signal
- Contrastive training uses negative sampling loss with interference energy as score
- The existing `wave_embeddings_plan.md` contains an earlier iteration of the architecture with phase instead of harmonic-count; the Roadmap.md reflects the current direction
