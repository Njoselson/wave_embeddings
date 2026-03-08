# Wave Embeddings Roadmap

## The Idea

Standard transformers represent tokens as dense vectors with hundreds or thousands of dimensions. Wave Embeddings instead represents each token as a small set of interfering complex exponential waves — each described by just 2 learnable scalars (frequency, amplitude).

**Per-token representation:** 3 waves x 2 parameters = 6 trainable scalars per token

Each wave is a complex exponential:
```
z_k(t) = A_k * exp(j * 2*pi * f_k * t)
```

A token's signal is the sum of its 3 waves. Similarity between tokens is measured via **wave interference energy** — when two tokens share frequencies, their waves constructively interfere, producing higher energy. The cross-term `2*A1*A2*sinc(delta_f)` is the similarity signal.

## Why This Might Work

- Complex exponentials have a smooth, convex loss landscape for frequency — unlike sin(wt) which has dense local minima (Hayes et al. 2022)
- Fourier space is the natural domain for wave interference — composition is just addition
- 6 params/token vs 768+ is orders of magnitude fewer parameters
- Wave interference energy provides a physically-motivated, differentiable similarity metric
- If the universal-frequency hypothesis holds, multilingual transfer comes for free

## Phase 0: Foundations & Validation (COMPLETE)

Architecture evolution through 4 iterations on SST-2 sentiment classification:

- **v1: Real sinusoids + harmonics** — 7 waves x 3 params (f, A, H) = 21 params/token. Frequencies did not learn due to non-convex sin() landscape. Amplitudes dominated.
- **v2: Phase-shifted sinusoids** — Added learnable phase. Same frequency learning problem.
- **v3: Complex exponentials** — 3 waves x 2 params (f, A) = 6 params/token. Frequencies actually learned! Complex exponential provides smooth gradient path through the complex plane.

Key findings:
- [x] Complex exponentials enable frequency learning (real sinusoids do not)
- [x] SST-2 validation accuracy: 82.3% with only ~64K params
- [x] Frequencies move meaningfully during training with complex exponentials
- [x] Gradient flow verified through all parameters (13 tests passing)
- [x] Separate LR for frequencies (1e-4) vs amplitudes (1e-3) helps stability

Completed artifacts:
- `src/wave_embedding_v3.py` — WaveEmbeddingV3, WaveModelV3
- `src/tone_wave.py`, `src/wave_embedding.py`, `src/wave_model.py` — v1 (legacy)
- `src/tokenizer.py` — word-level tokenizer
- `experiments/train_sst2_v3.py` — v3 training script
- `tests/test_core.py` — gradient flow and shape tests

## Phase 1: Contrastive Training (Current)

**Goal:** Force each token's frequency to encode its semantic identity via skip-gram training with wave interference energy.

Classification (Phase 0) doesn't force meaningful frequency differentiation — contrastive training does.

- [x] Wave interference energy similarity function (`src/wave_contrastive.py`)
- [x] Skip-gram dataset with negative sampling (`src/skipgram_dataset.py`)
- [x] Contrastive training script on English Wikipedia (`experiments/train_contrastive.py`)
- [x] WordSim-353 / SimLex-999 evaluation (`experiments/eval_similarity.py`)
- [x] Tests for interference energy, dataset, gradient flow (`tests/test_contrastive.py`)
- [ ] Run training on ~10K Wikipedia articles
- [ ] Evaluate: target Spearman rho > 0.3 on WordSim-353
- [ ] Verify frequency movement is greater than in classification
- [ ] Inspect: do semantically related words share frequencies?

## Phase 2: Multilingual & Universal Frequencies

**Goal:** Test whether concept-frequencies are language-invariant.

- [ ] Train on multilingual Wikipedia (English + Spanish + Mandarin + Arabic)
- [ ] Cross-lingual evaluation:
  - Train on language A, test similarity judgments on language B
  - Do translation-equivalent words converge to similar frequencies?
  - Measure alignment without explicit supervision
- [ ] If frequencies align cross-lingually, strong evidence for universal-frequency hypothesis

## Phase 3: Scaling & Applications

**Goal:** Push toward practical use cases.

- [ ] Sequence-level tasks: document classification, NLI using interference patterns
- [ ] Wave-space attention: frequency-domain gating as attention mechanism
- [ ] Benchmark compute/memory vs transformer baselines
- [ ] Investigate: can wave representations compress/distill transformer knowledge?

## Open Questions

### Resolved
1. **Harmonic differentiability** — sigmoid envelope works but is moot; v3 dropped harmonics entirely
2. **Frequency learning** — requires complex exponentials; real sinusoids have non-convex landscape
3. **Architecture choice** — v3 (complex exponentials, no harmonics) validated as best approach

### Open
1. **Frequency collapse:** Will contrastive training maintain frequency diversity, or will tokens collapse? May need diversity regularization.
2. **Optimal number of waves:** 3 works for classification; contrastive training may benefit from more.
3. **Composition beyond addition:** Negation, binding, and other non-additive semantics.
4. **Positional encoding:** Waves encode position implicitly via phase — is this sufficient?
5. **Scaling laws:** How does performance scale with number of waves and sample points?
6. **Interference energy scaling:** Does the energy metric need normalization for varying-length sequences?

## Technical Notes

- Complex exponentials `A*exp(j*2pi*f*t)` provide smooth loss landscape for frequency learning (Hayes et al. 2022)
- `torch.fft.rfft` and `torch.fft.irfft` are fully differentiable
- Frequency gradients are volatile — use Adam with low LR (1e-4) or separate param groups
- Interference energy: `E = A1^2 + A2^2 + 2*A1*A2*sinc(delta_f)` — the cross-term is the similarity signal
- Negative sampling loss: `L = -log(sigma(E_pos)) - sum(log(sigma(-E_neg)))` — standard word2vec objective with energy as score
