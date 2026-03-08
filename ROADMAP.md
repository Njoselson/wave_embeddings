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

## Phase 1.5: Closing the Benchmark Gap (Current)

**Goal:** Improve WordSim-353 from ~0 to >0.3 Spearman rho before moving to new tasks.

The v6 experiment showed the architecture *can* learn semantics but doesn't generalize. This phase addresses the likely causes.

### Increase model capacity
- [ ] Increase params/token: try 10-20 params (more waves or higher-dim amplitudes)
- [ ] Try 5-10 waves per band instead of 1 fundamental + harmonics
- [ ] Experiment with per-harmonic phase (currently phase is shared across harmonics)

### Improve training signal
- [ ] Train on larger corpus: full English Wikipedia (~100M tokens) instead of WikiText-2 (2M)
- [ ] More epochs with learning rate scheduling (cosine or reduce-on-plateau)
- [ ] Hard negative mining: sample negatives closer in frequency space for stronger gradients
- [ ] Increase window size to capture broader context

### Fix regularization
- [ ] Anneal frequency diversity weight: full strength early, zero by final third of training
- [ ] Or replace with softer constraint: minimize overlap of *unrelated* tokens only
- [ ] Experiment with no diversity regularizer at all — let contrastive loss handle it

### Training dynamics
- [ ] Warm up frequency learning rate (currently constant 3e-4)
- [ ] Try AdamW or LAMB optimizer for better generalization
- [ ] Add gradient noise to escape local minima in frequency space
- [ ] Monitor per-pair similarity trajectories to detect and prevent late-training degradation

### Evaluation
- [ ] Fix SimLex-999 parsing (format mismatch in current eval)
- [ ] Add MEN-3000 and RareWords benchmarks
- [ ] Per-category analysis: which semantic relationships does wave interference capture vs miss?

## Phase 2: Language Modeling with Wave Embeddings

**Goal:** Use wave embeddings for next-token prediction without requiring O(T * V * H^2) scoring.

The v6 LM (`train_lm_v6.py`) computed full wave interference between every position and all vocab tokens — this was O(T * V * H^2) per forward pass and OOM'd on 8GB GPUs even with V=3000.

### Approach A: Wave embeddings as drop-in layer
- [ ] Freeze trained wave embeddings, project to d-dim vector via small MLP
- [ ] Feed into standard small transformer (2-4 layers, d=128-256)
- [ ] Compare perplexity vs randomly initialized embeddings of same dimensionality
- [ ] This tests: do 5 learned wave params encode more than 5 random params?

### Approach B: Efficient wave LM
- [ ] Approximate scoring: only compute interference with top-K candidates from frequency-space ANN
- [ ] Or: use FFT-based fast interference computation (batch all vocab in Fourier domain)
- [ ] Gradient checkpointing to reduce memory from autograd graph
- [ ] Target: train on WikiText-2 with V=10K in <16GB VRAM

### Approach C: Wave-space autoregressive model
- [ ] Context representation: decayed running wave (already implemented in WaveLMv6)
- [ ] Prediction: learn a mapping from running wave → next token's wave params
- [ ] Avoids scoring all vocab — predicts wave params directly, then nearest-neighbor lookup

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
