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

## Phase 1: Contrastive Training (COMPLETE)

**Goal:** Force each token's frequency to encode its semantic identity via skip-gram training with wave interference energy.

Architecture evolved to v6: multi-scale (slow + fast bands), 7 harmonics per band, phase-aware interference, frequency diversity regularization. 5 params/token (freq_slow, freq_fast, amplitude, phase, scale_mix) + 2 global decay params.

- [x] Wave interference energy similarity function (`src/wave_embedding_v6.py`)
- [x] Skip-gram dataset with negative sampling (`src/skipgram_dataset.py`)
- [x] O(1) negative sampling via unigram table (word2vec-style)
- [x] GPU-side data loading for 97% utilization (`experiments/train_contrastive_v6.py`)
- [x] WordSim-353 / SimLex-999 evaluation (`experiments/eval_v6.py`)
- [x] 4-panel visualization: waveforms, spectra, similarity heatmap, frequency landscape (`experiments/visualize_waves.py`)
- [x] Trained on WikiText-2 (2.08M tokens, 10K vocab, 20 epochs)
- [x] Verified frequency movement (freq_slow Δ=0.11, freq_fast Δ=0.10)
- [x] Discovered phase initialization bug: zero init → zero gradient saddle point
- [x] Phase fix (randn * 0.3) enabled gender pair learning (man/woman: +0.84)

**Results:**
- Hand-picked pairs: good/great +0.84, man/woman +0.84 (with phase)
- WordSim-353 Spearman rho: -0.08 (no phase), -0.06 (with phase) — does not generalize
- 50K params total vs 3M for Word2Vec-300d (60x compression)
- Full results in `RESULTS.md`

**Key lessons:**
- Phase is essential — zero init is a saddle point, random init breaks symmetry
- Frequency diversity regularizer is double-edged: prevents collapse early but hurts semantics late
- 5 params/token learns a few strong relationships but doesn't generalize across benchmarks
- Skip-gram on WikiText-2 may be insufficient signal for such compressed representations

## Design Principles

Every experiment going forward must be evaluated against these four criteria:

1. **Training complexity** — Fewer hyperparameters, fewer optimizer groups, fewer interacting loss terms. v5 (2 params, 1 objective, 1 LR split) generalized better than v6 (5 params, 2 losses, 4 optimizer groups). Simplicity wins.

2. **Objective function smoothness** — The loss landscape must be convex or near-convex in the parameters we care about. `sinc(x)` oscillates beyond the central lobe, creating local minima. `cos(phase)` has a saddle point at zero. LM cross-entropy is smoother than contrastive skip-gram with sinc scoring — this may explain why v5 LM (rho=0.23) beat v6 contrastive (rho=-0.06).

3. **Learnability** — No conflicting gradients. v6's diversity regularizer fights the contrastive loss: one pushes frequencies apart, the other pulls co-occurring tokens together. When two loss terms compete, neither wins cleanly. A single well-designed objective is better than two that interfere.

4. **Inference speed** — Wave embeddings are already fast: table lookup + sinc evaluations, no matrix multiplies. Must preserve this. The LM scoring bottleneck (O(T * V * H²)) must be solved without adding dense layers that negate the compression advantage.

**Implication:** The next experiment should be the *simplest possible* change that addresses the v5→v6 regression, not the most feature-rich.

## Phase 1.5: Spectral State Models (Current)

### v7: Spectral Perturbations with MLM (COMPLETE)

Fundamental rethink: tokens are spectral PERTURBATIONS to a running state, not static waves. Bidirectional MLM with cross-entropy loss.

- [x] Architecture: `SpectralState`, `SpectralEmbedding`, `SpectralStateLM` (`src/wave_embedding_v7.py`)
- [x] Efficient scoring via cos/sin decomposition: `(B, S*K) @ (S*K, V)` matmul
- [x] MLM training on WikiText-2 (`experiments/train_spectral_v7.py`)
- [x] Capacity scaling: K=4 (24p/tok), K=8 (48p/tok), K=16 (96p/tok)
- [x] PPL 706 — 2.2x better than v5's 1,563
- [x] Strong word pairs: war/battle +0.90, man/woman +0.42
- [x] WS-353 eval (`experiments/eval_v7.py`)

**Key findings:**
- Capacity doesn't matter: K=4/8/16 all hit PPL 706. The bottleneck is the linear state evolution, not embedding dimension.
- Word pairs peak at epoch 10-15 then degrade (MLM overfitting)
- WS-353 still negative — function words cluster near +1.0 similarity
- Multi-scale decay self-organizes: long=0.96, mid=0.63, short=0.12

### v7.1: Gated State Updates (IN PROGRESS)

The linear recurrence `state = decay * state + delta` is the expressiveness bottleneck. Content-dependent gating adds selective memory.

- [x] Architecture: `GatedSpectralStateLM` (`src/wave_embedding_v7_1.py`)
- [x] Gate = sigmoid(token_bias + sensitivity * coherence(state, perturbation))
- [x] Per-token gate_bias: function words learn to pass through, content words learn to overwrite
- [x] Training script (`experiments/train_spectral_v7_1.py`)
- [ ] Train on WikiText-2 and compare PPL against v7
- [ ] Monitor gate bias divergence (function vs content words)
- [ ] Does PPL break below 700?

### v7.2: Complex-Native Tensors (PLANNED)

The cos/sin decomposition is implicitly complex arithmetic. Making it explicit simplifies the code and enables native complex matmul.

- [ ] State as complex tensor: `z = amp * exp(j * phase)` — (B, S, K) complex64
- [ ] Perturbation as complex: `delta = delta_amp * exp(j * delta_phase)`
- [ ] State update: complex multiply/add instead of separate amp/phase arithmetic
- [ ] Scoring: `Re(state @ conj(vocab).T)` — single complex matmul
- [ ] Gated update: `state = gate * state + (1-gate) * delta` works directly on complex tensors
- [ ] Potential benefit: cleaner gradients through complex autograd, simpler code

### Future: Parallel State Computation

The sequential state loop (O(T) serial steps) is a speed bottleneck. The linear recurrence `state = a * state + b` can be parallelized via prefix scan in O(log T). The gated version (v7.1) makes this harder but Mamba showed it's possible with input-dependent state transitions.

- [ ] Implement parallel scan for linear recurrence (v7)
- [ ] Investigate selective scan for gated recurrence (v7.1)
- [ ] Benchmark: sequential vs parallel on GPU for T=64, 128, 256

## Phase 3: Multilingual & Universal Frequencies

**Goal:** Test whether concept-frequencies are language-invariant.

- [ ] Train on multilingual Wikipedia (English + Spanish + Mandarin + Arabic)
- [ ] Cross-lingual evaluation:
  - Train on language A, test similarity judgments on language B
  - Do translation-equivalent words converge to similar frequencies?
  - Measure alignment without explicit supervision
- [ ] If frequencies align cross-lingually, strong evidence for universal-frequency hypothesis
- [ ] Requires Phase 1.5 success first — need >0.3 rho on monolingual before testing cross-lingual

## Phase 4: Scaling & Applications

**Goal:** Push toward practical use cases.

- [ ] Sequence-level tasks: document classification, NLI using interference patterns
- [ ] Wave-space attention: frequency-domain gating as attention mechanism
- [ ] Benchmark compute/memory vs transformer baselines at scale
- [ ] Investigate: can wave representations compress/distill transformer knowledge?
- [ ] Edge deployment: wave embeddings on mobile/IoT (50K params fits in L1 cache)

## Open Questions

### Resolved
1. **Harmonic differentiability** — sigmoid envelope works but is moot; v3 dropped harmonics entirely
2. **Frequency learning** — requires complex exponentials; real sinusoids have non-convex landscape
3. **Architecture choice** — v3 (complex exponentials, no harmonics) validated as best approach
4. **Frequency collapse** — diversity regularizer prevents it, but needs annealing to avoid hurting semantics
5. **Phase learning** — requires non-zero initialization; zero init is a saddle point with zero gradient
6. **LM feasibility** — cos/sin decomposition solves the O(T*V*H^2) scoring bottleneck via matmul
7. **Optimal params/token for LM** — capacity is NOT the bottleneck; K=4/8/16 all hit same PPL. The linear state evolution is the limiter.
8. **LM vs skip-gram** — MLM cross-entropy (v7) dramatically outperforms skip-gram contrastive (v6)

### Open
1. **Does gating break the PPL plateau?** v7.1 adds content-dependent gates — will PPL drop below 700?
2. **Composition beyond addition:** Negation, binding, and other non-additive semantics. Gating is a step toward this.
3. **Positional encoding:** Spectral state carries implicit position via decay. Is this sufficient for longer sequences?
4. **Scaling laws:** Does WikiText-2 (2M tokens) saturate the model? Would WikiText-103 help?
5. **Benchmark gap:** Strong word pairs but negative WS-353. Is `spectral_similarity` (raw perturbation comparison) the wrong metric? Should similarity be measured via contextual state instead?
6. **Complex-native representation:** Does explicit complex arithmetic improve gradient flow or training stability?

## Technical Notes

- Complex exponentials `A*exp(j*2pi*f*t)` provide smooth loss landscape for frequency learning (Hayes et al. 2022)
- `torch.fft.rfft` and `torch.fft.irfft` are fully differentiable
- Frequency gradients are volatile — use Adam with low LR (3e-4) or separate param groups
- Interference energy: `E = sum_ij A_i * A_j * sinc(2*df_ij) * cos(dphi_ij)` — phase-aware cross-terms
- Negative sampling loss: `L = -log(sigma(E_pos)) - sum(log(sigma(-E_neg)))` — standard word2vec objective with energy as score
- Phase init: must be non-zero (randn * 0.3) to avoid saddle point where cos gradient is zero
- GPU data loading: for small models, move all data to GPU tensors and sample with `torch.randint` — eliminates DataLoader overhead entirely
- Vast.ai: RTX 3060 Ti at $0.063/hr is sufficient for contrastive training; LM needs >8GB VRAM
