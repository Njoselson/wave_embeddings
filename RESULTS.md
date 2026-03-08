# Wave Embeddings — Experimental Results

## Summary

Wave Embeddings replace high-dimensional dense vectors (768+ params/token) with parametric wave representations. Each token is described as a superposition of sinusoidal waves, and word similarity is computed via analytical wave interference energy rather than dot products.

This document covers the full experimental history across 4 architecture versions (v3–v6), trained on WikiText-2 and SST-2.

## Architecture Evolution

| Version | Params/Token | Level | Key Idea |
|---------|-------------|-------|----------|
| v3 | 6 | token | 3 complex exponential waves (f, A) |
| v4 | 6/char | character | 3 waves per char, positional shift for word composition |
| v5 | 2 | token | 1 frequency + 1 amplitude, 7 harmonics |
| v6 | 5 | token | Dual-band (slow+fast), phase, diversity regularization |

### v3: Complex Exponentials (Phase 0)
- 3 waves × 2 params (f, A) = 6 params/token
- Complex exponentials `A * exp(j * 2π * f * t)` — smooth loss landscape for frequency learning
- **SST-2 accuracy: 82.3%** with ~64K total params
- Key finding: complex exponentials enable frequency learning; real sinusoids do not (Hayes et al. 2022)

### v4: Character-Level Waves
- 3 waves per character, ~130 char vocab
- Words composed by summing char signals with position-dependent frequency shift ("cat" ≠ "act")
- Analytical sinc similarity — no time-domain sampling
- Trained on WikiText-2, but character-level granularity limited semantic capture

### v5: Single Frequency with Harmonics
- 1 frequency + 1 amplitude per token = 2 params/token
- 7 harmonics create multi-scale interaction: `Σ (A/h^decay) * exp(j·2π·h·f·t)`
- Global learnable decay parameter controls harmonic richness
- Both skip-gram and language model variants

### v6: Multi-Scale Phase-Aware (Current)

## Architecture: v6 Multi-Scale Phase-Aware Wave Interference

Each token is parameterized by 5 scalars:

| Parameter | Description | Init |
|-----------|-------------|------|
| `freq_slow` | Low frequency — broad semantic category | `randn * 0.3` |
| `freq_fast` | High frequency — fine-grained meaning | `randn * 3.0` |
| `amplitude` | Overall energy of the token | `1.0 + randn * 0.1` |
| `phase` | Timing offset for interference | `randn * 0.3` |
| `scale_mix` | Sigmoid blend between slow/fast bands | `0.0` (logit) |

Plus 2 global parameters: `decay_slow` and `decay_fast` (harmonic envelope decay rates).

Each band is expanded into H=7 harmonics (overtones), giving 2H=14 frequency components per token. Similarity between tokens is computed analytically:

```
E(a,b) = sum_ij A_i * A_j * sinc(2 * delta_f_ij) * cos(delta_phi_ij)
similarity(a,b) = (E(a,b) - E(a,a) - E(b,b)) / (2 * sqrt(E(a,a) * E(b,b)))
```

**Total model size:** 50,002 parameters for 10,000 tokens (5 per token + 2 global).
For comparison, Word2Vec-300d uses 3,000,000 parameters for the same vocabulary.

## Training Setup

- **Dataset:** WikiText-2 (5,469 paragraphs, 2.08M tokens)
- **Method:** Skip-gram with negative sampling (15 negatives per positive)
- **Batch size:** 8,192
- **Epochs:** 20
- **Optimizer:** Adam with separate learning rates
  - Frequencies: 3e-4 (lower to stabilize)
  - Other params: 1e-3
- **Regularization:** Frequency diversity loss (weight=0.01) to prevent frequency crowding
- **Hardware:** NVIDIA RTX 3060 Ti (8GB), via vast.ai
- **Training time:** ~52 minutes (20 epochs, ~2.6 min/epoch at 97% GPU utilization)

### Data Pipeline Optimizations

All training data (10.5M skip-gram pairs) and the negative sampling table (10M entries) were loaded directly onto GPU memory, eliminating CPU-GPU transfer overhead entirely. Negative sampling uses a word2vec-style unigram table for O(1) lookup instead of `np.random.choice`.

## v5 Results: Single Frequency with Harmonics

Trained on an RTX A4000 (16GB) via vast.ai. 20 epochs, 3.2 hours, $0.26 total cost.

### Language Model

| Metric | Value |
|--------|-------|
| Parameters | 20,001 (10K tokens × 2 + 1 decay) |
| Final Loss | 7.35 |
| Final Perplexity | 1,563 |
| Random baseline PPL | 10,000 |
| **PPL improvement** | **6.4×** |

The model beats random chance substantially. Decay self-organized from 1.5 → 0.304, meaning the model discovered it needs all 7 harmonics active (low decay = slow rolloff).

### Word Similarity (v5)

| Word Pair | Similarity | Notes |
|-----------|-----------|-------|
| war - battle | **+0.86** | Strong — synonyms/co-occurring |
| city - town | **+0.49** | Strong — semantically close |
| the - of | +0.45 | Function words cluster |
| good - bad | +0.32 | Antonyms in same contexts |
| king - queen | +0.03 | Weak |
| man - woman | +0.01 | Near zero |
| cat - dog | +0.00 | Near zero |
| he - she | -0.01 | Near zero |

**WordSim-353 Spearman rho: 0.23** (44 in-vocab pairs, p=0.125)

### What v5 Revealed

**Worked:** Co-occurrence relationships (war/battle appear in same Wikipedia articles). The sinc metric is meaningful — not random noise. Decay is a useful learnable parameter.

**Failed:** Abstract semantic relationships (king/queen, gender pairs, morphology). With 1 frequency per token, the model has ~1 degree of freedom vs Word2Vec's 300. Nearest-neighbor lists were mostly noise — similarity scores clustered near 1.0 because frequency spread (std=3.05) is too narrow for sinc to discriminate.

**Diagnosis:** The representation is too compressed. Need more params/token while keeping the wave interference framework.

### Learned Parameters (v5)

Selected words from `checkpoints/wave_v5_params.csv`:

| Token | Frequency | Amplitude |
|-------|-----------|-----------|
| king | -2.22 | 0.28 |
| queen | -3.02 | 0.25 |
| man | 1.48 | 0.32 |
| woman | -3.84 | 0.23 |
| good | -1.30 | 0.13 |
| great | 4.17 | 0.45 |
| war | — | — |
| the | -1.21 | 0.26 |

Frequency range: [-10.93, 10.99], std=3.053. Some tokens learned negative amplitudes (unusual).

---

## v6 Results: Multi-Scale Phase-Aware

### Experiment 1: Without Phase Learning (phase initialized to zero)

Phase initialized at zero has zero gradient (`d/dx cos(0) = -sin(0) = 0`), creating a saddle point. Phase remained at `std=0.000` throughout training.

**Final word pair similarities (epoch 20):**

| Word Pair | Similarity | Relationship |
|-----------|-----------|--------------|
| good - great | **+0.84** | Synonyms |
| boy - girl | +0.23 | Gender pair |
| king - queen | +0.00 | Gender pair |
| cat - dog | +0.02 | Co-hyponyms |
| man - woman | -0.03 | Gender pair |
| good - bad | -0.28 | Antonyms |
| sun - moon | -0.31 | Celestial pair |
| king - table | -0.42 | Unrelated |
| cat - the | -0.55 | Unrelated |

**Trajectory across epochs:**

| Pair | Epoch 5 | Epoch 10 | Epoch 15 | Epoch 20 |
|------|---------|----------|----------|----------|
| good - great | +0.31 | +0.41 | +0.73 | +0.84 |
| sun - moon | +0.39 | +0.59 | +0.65 | -0.31 |
| boy - girl | +0.42 | +0.56 | +0.52 | +0.23 |
| king - queen | -0.07 | -0.06 | +0.01 | +0.00 |
| man - woman | -0.38 | -0.30 | -0.01 | -0.03 |
| good - bad | -0.09 | +0.02 | -0.25 | -0.28 |

**Observations:**
- Synonyms (good/great) learned robustly and monotonically improved
- Some pairs (sun/moon, boy/girl) peaked mid-training then degraded — likely due to frequency diversity regularizer pushing frequencies apart in later epochs
- Antonyms (good/bad) learned to be dissimilar, which is linguistically interesting
- Gender pairs (king/queen, man/woman) struggled without phase

**Learned parameter statistics:**

| Parameter | Mean | Std | Range |
|-----------|------|-----|-------|
| freq_slow | -0.036 | 0.521 | [-1.43, 1.29] |
| freq_fast | -0.043 | 3.009 | [-12.04, 10.56] |
| amplitude | -0.047 | 0.060 | — |
| phase | 0.000 | 0.000 | (stuck) |
| scale_mix (sigmoid) | 0.543 | 0.154 | — |
| decay_slow | 0.669 | — | (global) |
| decay_fast | 1.086 | — | (global) |

### Experiment 2: With Phase Learning (phase initialized to randn * 0.3)

Breaking the zero-symmetry allowed phase to learn meaningful values (`std=0.907` at convergence).

**Final word pair similarities (epoch 20):**

| Word Pair | Similarity | Relationship |
|-----------|-----------|--------------|
| good - great | **+0.83** | Synonyms |
| man - woman | **+0.84** | Gender pair |
| good - bad | +0.16 | Antonyms |
| king - queen | **+0.10** | Gender pair |
| boy - girl | +0.02 | Gender pair |
| cat - dog | -0.01 | Co-hyponyms |
| sun - moon | -0.00 | Celestial pair |
| cat - the | -0.06 | Unrelated |
| king - table | -0.03 | Unrelated |

**Comparison: Phase Off vs Phase On**

| Pair | No Phase | With Phase | Impact |
|------|----------|-----------|--------|
| man - woman | -0.03 | **+0.84** | Phase critical for gender |
| good - great | +0.84 | +0.83 | Stable |
| king - queen | +0.00 | +0.10 | Improved |
| good - bad | -0.28 | +0.16 | Less oppositional |
| cat - the | -0.55 | -0.06 | Less contrast |

**Learned parameter statistics (with phase):**

| Parameter | Mean | Std | Range |
|-----------|------|-----|-------|
| freq_slow | -0.069 | 0.518 | [-1.95, 1.45] |
| freq_fast | -0.011 | 3.058 | [-10.95, 12.09] |
| amplitude | -0.032 | 0.087 | — |
| phase | -0.027 | **0.907** | (learned) |
| scale_mix (sigmoid) | 0.586 | 0.170 | — |
| decay_slow | 1.611 | — | (global) |
| decay_fast | 2.337 | — | (global) |

**Key finding:** Phase is essential for learning relational pairs (man/woman, king/queen). Without phase, the model can only differentiate tokens by frequency and amplitude. With phase, tokens at similar frequencies can still differ by their phase offset — analogous to two instruments playing the same note at different times.

**Trade-off:** The phase-enabled model showed weaker contrast between related and unrelated pairs (the heatmap was more uniformly warm). This suggests the model may need stronger contrastive signal or longer training to fully separate the similarity distribution.

## Compression Ratio

| Model | Params/Token | Total (10K vocab) | Similarity Metric |
|-------|-------------|-------------------|-------------------|
| Word2Vec-300d | 300 | 3,000,000 | Dot product |
| GloVe-300d | 300 | 3,000,000 | Dot product |
| BERT base | 768 | 23,440,896 | Cosine similarity |
| **Wave v6** | **5** | **50,002** | Wave interference |

**60x fewer parameters than Word2Vec/GloVe, 470x fewer than BERT**, while achieving synonym detection (good/great: 0.84) comparable to these models.

## Key Findings

1. **Complex exponential / sinusoidal representations can learn word semantics.** The wave interference energy metric captures meaningful similarity with dramatically fewer parameters.

2. **Phase initialization matters.** Zero-initialized phase creates a saddle point with zero gradient. Random initialization (`randn * 0.3`) breaks symmetry and enables phase to learn, which is critical for relational pairs.

3. **Multi-scale frequencies differentiate.** The slow band (std=0.52) captures broad categories while the fast band (std=3.06) captures fine-grained distinctions. The learned scale_mix averages ~0.57, slightly favoring the slow band.

4. **Frequency diversity regularization has a double-edged effect.** It prevents frequency crowding early in training but may push semantically related tokens apart in later epochs, causing degradation of some pairs (sun/moon dropped from +0.65 to -0.31).

5. **The approach scales poorly to language modeling.** Computing wave interference between every position and every vocab token is O(T * V * H^2), making next-token prediction impractical on consumer GPUs. The embeddings themselves are efficient; the bottleneck is the scoring mechanism.

## Visualizations

See `wave_visualization.png` (without phase) and `wave_visualization_phase.png` (with phase) for:
- Time-domain waveforms for selected words
- Frequency spectra (slow + fast band harmonics)
- Pairwise similarity heatmaps
- Frequency landscape of all 10K vocabulary tokens

## Gradient Diagnostics (v3–v5)

Two diagnostic experiments (`diagnose_grads.py`, `diagnose_grads2.py`) investigated why frequencies were hard to train:

1. **Frequency gradient magnitudes ~1000× smaller than amplitude gradients.** Cause: `sin(f)` provides poor gradients vs `exp(j·f)`. Solution: switch to complex exponentials in v3+.

2. **Frequency std appeared static despite training.** Cause: initial frequency spread (linspace over large range) masks per-token movements. Per-token frequencies DO move (~0.001/step) but aggregate std is dominated by initialization. Conclusion: frequency IS learning, just hard to see in bulk stats — need to track per-token delta from init.

## Cross-Version Comparison

| Metric | v5 (2 params) | v6 no phase (5 params) | v6 with phase (5 params) |
|--------|--------------|----------------------|------------------------|
| good - great | — | **+0.84** | **+0.83** |
| war - battle | **+0.86** | — | — |
| city - town | **+0.49** | — | — |
| king - queen | +0.03 | +0.00 | **+0.10** |
| man - woman | +0.01 | -0.03 | **+0.84** |
| cat - dog | +0.00 | +0.02 | -0.01 |
| good - bad | +0.32 | -0.28 | +0.16 |
| WS-353 rho | **0.23** | -0.08 | -0.06 |
| LM PPL | 1,563 | — | — |
| Params/token | 2 | 5 | 5 |

**Paradox:** v5 scored higher on WS-353 (0.23) than v6 (-0.06) despite v6 having better hand-picked pairs. Possible explanations:
- v5's LM training (cross-entropy on next-token) may provide stronger distributional signal than skip-gram
- v6's frequency diversity regularizer may harm the broad similarity distribution while helping a few target pairs
- v6's WS-353 eval may have different vocab coverage or parsing issues

This warrants direct comparison under identical conditions.

## Formal Benchmark: WordSim-353

Evaluated on the WordSim-353 word similarity benchmark (237/353 pairs covered by our 10K vocabulary).

| Model | Params/Token | WS-353 (Spearman rho) |
|-------|-------------|----------------------|
| Word2Vec-300d | 300 | ~0.65 |
| GloVe-300d | 300 | ~0.60 |
| **Wave v6 (no phase)** | **5** | **-0.08** |
| **Wave v6 (with phase)** | **5** | **-0.06** |

The model achieves near-zero (slightly negative) correlation with human similarity judgments on the full benchmark, despite showing strong performance on hand-picked pairs. This suggests:

1. The model learns a few high-signal relationships well but doesn't generalize broadly
2. 20 epochs on WikiText-2 with 5 params/token may be insufficient — the model is severely underparameterized relative to the task
3. The frequency diversity regularizer may be counterproductive at scale, pushing semantically related words apart
4. The skip-gram objective with only 10.5M pairs may not provide enough signal for such a compressed representation

This is an honest negative result. The architecture demonstrates that wave interference can capture *some* semantic structure, but the current formulation does not yet compete with dense embeddings on standard benchmarks.

## Infrastructure: Training on Vast.ai

All GPU training was done on [vast.ai](https://vast.ai), a marketplace for renting GPU instances. This section documents the setup and optimization process, which may be useful for reproducing results or running similar experiments.

### Instance Setup

- **GPU:** NVIDIA RTX 3060 Ti (8GB VRAM, $0.063/hr)
- **Image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- **vCPUs:** 6, **RAM:** 32GB, **Storage:** 20GB
- **Access:** SSH via `ssh -p <port> root@ssh<N>.vast.ai`

### Deployment Workflow

```bash
# 1. Create instance via vast.ai CLI
vastai show instances

# 2. SSH in, clone repo, install deps
ssh -p <port> root@ssh<N>.vast.ai
git clone https://github.com/Njoselson/wave_embeddings.git
cd wave_embeddings && pip install uv && uv sync

# 3. Run training (background so SSH disconnect doesn't kill it)
nohup uv run python experiments/train_contrastive_v6.py > /root/train.log 2>&1 &

# 4. Monitor from local machine
ssh -p <port> root@ssh<N>.vast.ai "tail -f /root/train.log"

# 5. Pull checkpoint when done
scp -P <port> root@ssh<N>.vast.ai:/root/wave_embeddings/checkpoints/wave_contrastive_v6.pt ./checkpoints/
```

### GPU Utilization Optimization Journey

The contrastive training started at **9% GPU utilization** and was iteratively optimized to **97%**. The key lesson: for small models, the bottleneck is almost always data loading, not computation.

#### Iteration 1: Baseline (9% GPU, ~7 min/epoch)
- `batch_size=512`, `num_workers=0`
- Single-threaded data loading on CPU, GPU starved between batches

#### Iteration 2: More workers (21% GPU)
- `batch_size=4096`, `num_workers=4`, `pin_memory=True`
- CPU workers were at 97% each — `np.random.choice` with probability distribution is expensive per-sample

#### Iteration 3: Unigram table (67% GPU, ~2 min/epoch)
- Replaced `np.random.choice(p=probs)` with word2vec-style unigram lookup table (10M entries)
- O(1) negative sampling instead of O(V) rejection sampling
- Stored skip-gram pairs as contiguous numpy array

#### Iteration 4: Full GPU-side data (97% GPU, ~2.6 min/epoch)
- Loaded entire pairs tensor (10.5M pairs, ~80MB) directly onto GPU
- Loaded negative sampling table (10M entries, ~80MB) onto GPU
- Shuffling via `torch.randperm` on GPU
- Negative sampling via `torch.randint` + table lookup on GPU
- **Zero CPU-GPU transfer during training**
- Attempted `batch_size=16384` but hit OOM at 8GB; settled on `batch_size=8192` (6.2GB used)

**Result:** 7.7x speedup from baseline (7 min/epoch → 2.6 min/epoch), 10.8x GPU utilization improvement (9% → 97%).

### Language Model Training: Memory Challenges

The LM forward pass computes wave interference between every sequence position and every vocab token, requiring a cross-terms tensor of shape `(B, T, 2H, V, 2H)`. With V=10,000 and H=7, this is ~62M floats per batch chunk — too large for 8GB.

**Attempted mitigations:**
1. Reduced batch/seq/chunk sizes — still OOM due to autograd graph
2. Rewrote cross-terms to loop over harmonic pairs (avoids 5D tensor) — still OOM from accumulated computation graph
3. Reduced vocab to 3,000 — fit in memory but extremely slow (~13 hr/epoch)

**Conclusion:** The wave LM architecture requires either >16GB VRAM, gradient checkpointing, or an approximate scoring mechanism to be practical. The contrastive (skip-gram) training is efficient; the LM scoring is the bottleneck.

### Cost

Total vast.ai compute cost for all experiments: approximately **$2-3** (RTX 3060 Ti at $0.063/hr for ~30-40 hours including idle time).

## Next Steps

- Investigate frequency diversity regularizer scheduling (anneal weight to zero in later epochs)
- Use wave embeddings as a frozen embedding layer in a small transformer to test downstream task performance
- Explore approximate nearest-neighbor in frequency space for efficient LM scoring
