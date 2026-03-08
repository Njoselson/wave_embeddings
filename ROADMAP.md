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

## Phase 1.5: Closing the Benchmark Gap (Current)

**Goal:** Beat v5's WS-353 rho of 0.23. Target: >0.3.

### The v5 vs v6 paradox

v5 (2 params/token, LM objective) scored 0.23 on WS-353.
v6 (5 params/token, skip-gram + diversity reg) scored -0.06.

More params + more features = worse generalization. Three suspects:
1. **Objective**: LM cross-entropy (v5) vs skip-gram contrastive (v6)
2. **Regularizer**: v6's diversity loss fights the similarity objective
3. **Architecture complexity**: v6 has too many interacting parts

### Experiment plan: isolate one variable at a time

**Experiment A: v5 architecture + skip-gram** (isolate objective)
- [ ] Train v5 (2 params, 7 harmonics) with skip-gram objective (same as v6 training)
- [ ] No diversity regularizer
- [ ] Compare WS-353 rho against v5-LM (0.23) and v6-contrastive (-0.06)
- [ ] If rho drops: LM objective is the key ingredient
- [ ] If rho holds: v6's architecture/regularizer is the problem

**Experiment B: v6 architecture + LM objective** (isolate architecture)
- [ ] Train v6 (5 params, dual-band) with LM cross-entropy (same as v5 training)
- [ ] No diversity regularizer
- [ ] Compare against v5-LM (0.23)
- [ ] If rho improves: more params + LM = the path forward
- [ ] If rho drops: v6's added complexity (dual-band, phase) actively hurts

**Experiment C: v6 skip-gram without diversity loss** (isolate regularizer)
- [ ] Exact v6 contrastive setup but `freq_diversity_weight=0.0`
- [ ] Quickest test — only changes one config value
- [ ] If rho jumps: diversity loss was the problem all along

**Run order:** C first (cheapest), then A, then B.

### If the objective matters most (LM > skip-gram)

The sinc scoring function in skip-gram may be the issue:
- `sinc(2*Δf)` oscillates — two tokens with Δf=1.5 have *negative* sinc, creating false repulsion
- LM cross-entropy uses sinc indirectly through logits, softmax smooths the landscape
- Consider: replace sinc with Gaussian kernel `exp(-Δf²/σ²)` — monotonically decreasing, no oscillation, still differentiable

### If capacity matters (need more params)

Scale params/token gradually: 2 → 4 → 8 → 16. Don't jump to 5 with 3 new features at once.
- [ ] v5 with 2 frequencies + 2 amplitudes = 4 params (simplest capacity increase)
- [ ] v5 with 4 frequencies + 4 amplitudes = 8 params
- [ ] Plot WS-353 rho vs params/token — find the knee

### Evaluation improvements
- [ ] Fix SimLex-999 parsing (format mismatch in current eval)
- [ ] Ensure identical vocab coverage across v5 and v6 evals
- [ ] Per-category analysis: which semantic relationships does wave interference capture vs miss?

## Phase 2: Language Modeling with Wave Embeddings

**Goal:** Use wave embeddings for next-token prediction without requiring O(T * V * H²) scoring.

v5 LM achieved PPL 1,563 (6.4× over random) but OOM'd v6 on 8GB. The scoring mechanism must be efficient.

### Approach A: Wave embeddings as drop-in layer (simplest, tests embedding quality)
- [ ] Freeze trained wave embeddings, project to d-dim vector via small linear layer
- [ ] Feed into standard small transformer (2-4 layers, d=128-256)
- [ ] Compare perplexity vs randomly initialized embeddings of same total param count
- [ ] **Inference**: still fast — wave lookup + small linear projection + small transformer
- [ ] **Complexity**: minimal — just add a projection layer, no new objectives
- [ ] This answers: does the wave representation encode more information per parameter?

### Approach B: Gaussian kernel scoring (smoother objective, same architecture)
- [ ] Replace `sinc(2*Δf)` with `exp(-Δf²/σ²)` in the energy function
- [ ] Monotonically decreasing with frequency distance — no oscillatory local minima
- [ ] Same O(T*V*H²) cost but smoother loss landscape may allow larger learning rates
- [ ] σ is a learnable or tunable parameter

### Approach C: Predict wave params directly (fast inference)
- [ ] Context → predicted (f, A) for next token via small MLP
- [ ] Nearest-neighbor lookup in wave param space to find the token
- [ ] **Inference**: O(V) distance comparisons in low-dim space, or O(log V) with ANN index
- [ ] Avoids scoring all vocab through interference — decouples generation from wave math

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
6. **LM feasibility** — direct wave interference scoring is O(T*V*H^2), impractical; need approximate methods

### Open
1. **Optimal params/token:** 5 isn't enough for benchmarks. Where's the sweet spot between 5 and 300?
2. **Composition beyond addition:** Negation, binding, and other non-additive semantics.
3. **Positional encoding:** Waves encode position implicitly via phase — is this sufficient?
4. **Scaling laws:** How does performance scale with number of waves and corpus size?
5. **Interference energy scaling:** Does the energy metric need normalization for varying-length sequences?
6. **Why does generalization fail?** Hand-picked pairs work, benchmarks don't — is this overfitting to high-frequency co-occurrence pairs, or a fundamental limitation of the similarity metric?

## Technical Notes

- Complex exponentials `A*exp(j*2pi*f*t)` provide smooth loss landscape for frequency learning (Hayes et al. 2022)
- `torch.fft.rfft` and `torch.fft.irfft` are fully differentiable
- Frequency gradients are volatile — use Adam with low LR (3e-4) or separate param groups
- Interference energy: `E = sum_ij A_i * A_j * sinc(2*df_ij) * cos(dphi_ij)` — phase-aware cross-terms
- Negative sampling loss: `L = -log(sigma(E_pos)) - sum(log(sigma(-E_neg)))` — standard word2vec objective with energy as score
- Phase init: must be non-zero (randn * 0.3) to avoid saddle point where cos gradient is zero
- GPU data loading: for small models, move all data to GPU tensors and sample with `torch.randint` — eliminates DataLoader overhead entirely
- Vast.ai: RTX 3060 Ti at $0.063/hr is sufficient for contrastive training; LM needs >8GB VRAM
