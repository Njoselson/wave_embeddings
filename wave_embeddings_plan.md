# 🌊 Project Plan: Wave-Based Contextual Embeddings

This document outlines the architecture, specifications, and validation strategy for the novel Word Embedding system that uses trainable angular frequencies ($\omega$), magnitudes ($A$), and phases ($\phi$) combined with a Fast Fourier Transform (FFT) to generate a contextual frequency spectrum vector.

## I. System Architecture and Specifications

### 1. Core Model Components (Encoder)

| Component | Input | Output | Mechanism | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Word Factor** | Word Token $t$ | Signal Parameters $(\omega, A, \phi)$ | Lookup table based on vocabulary index. | $\mathbf{P}_t$: $[\omega_t, A_{t, 1}, \phi_{t, 1}, \ldots, A_{t, m}, \phi_{t, m}]$ |
| **Signal Generation** | $\mathbf{P}_t$ | Time-Domain Signal $\mathbf{S}_t$ (Length $L$) | $\mathbf{S}_t(n) = \sum_{k=1}^{m} A_{t, k} \cos(k \omega_t n + \phi_{t, k})$ | None (Fixed function) |
| **Spectral Conversion** | $\mathbf{S}_t$ | Raw Frequency Spectrum $\mathbf{X}_t$ (Dimension $D$) | Differentiable Fast Fourier Transform (FFT). | None (Fixed function) |
| **Contextual Layer** | $\mathbf{X}_t, \mathbf{X}_{\text{context}}$ | Contextual Embedding $\mathbf{E}_{\text{context}}$ (Dimension $D$) | Gated Modulation (e.g., using $\mathbf{W}_{\text{gate}}$ matrices). | $\mathbf{W}_{\text{gate}}, \mathbf{b}$ |

### 2. Hyperparameters & Dimensions

| Hyperparameter | Symbol | Initial Value | Notes / Rationale |
| :--- | :--- | :--- | :--- |
| **FFT Length** | $L$ | 1024 | Sets the effective output dimension $D \approx L/2 = 512$. |
| **Embedding Dimension** | $D$ | 512 | Standard size for comparison with models like Word2Vec. |
| **Harmonic Count** | $m$ | 10 | Each word has 21 (or $1+2m$) trainable parameters. |
| **Context Window** | $2n+1$ | 5 (2 before, 2 after) | Defines the range of neighbor spectra used in the Contextual Layer. |
| **Initial Learning Rate** | $\eta$ | $1 \times 10^{-4}$ | Start conservatively low due to the volatile $\omega$ gradients. |

## II. Loss Function Comparison and Selection

We will compare the two most viable training objectives, both aiming to train the underlying $\mathbf{P}_t$ parameters.

| Feature | Predictive Loss (Negative Sampling) | Reconstruction Loss (Autoencoder) |
| :--- | :--- | :--- |
| **Primary Goal** | Maximizes semantic similarity between true neighboring words, measured via dot product. | Minimizes error in reconstructing the center word $t_w$ from its compressed $\mathbf{E}_{\text{context}}$. |
| **Formula** | $\mathcal{L}_{\text{pred}} = -\log(\sigma(\mathbf{E}_{\text{target}}^T \mathbf{E}_{\text{pos}})) - \sum_{k} \log(\sigma(-\mathbf{E}_{\text{target}}^T \mathbf{E}_{\text{neg}, k}))$ | $\mathcal{L}_{\text{recon}} = -\sum y_i \log(P_{\text{reconstruction}, i})$ (Cross-Entropy) |
| **Required Decoder** | None (Dot product is the 'decoder'). | Required: A simple MLP to map $\mathbf{E}_{\text{context}}$ back to a $\text{Softmax}(V)$ distribution. |
| **Advantage** | **Simplicity & Speed.** Faster training by avoiding a large Softmax across the entire vocabulary $V$. | **Direct Signal & Stability.** Stronger, more direct gradient signal forcing maximal information encoding into $\mathbf{E}_{\text{context}}$. |
| **Risk** | Lower semantic quality than reconstruction due to the indirect similarity objective. | Computational cost is high due to the final Softmax over the large vocabulary $V$. |

**Validation Plan:** We must test both losses against the semantic quality benchmarks in Phase 2 to determine which training signal best tunes the $\omega, A, \phi$ parameters.

## III. Validation and Experimentation Plan

The validation process involves a two-phase approach to prove both stability and semantic quality.

### Phase 1: Trainability and Hyperparameter Validation

This phase focuses on the internal mechanics of the system using a small corpus (e.g., PTB).

| Test | Objective | Metric | Success Criteria |
| :--- | :--- | :--- | :--- |
| **A. Gradient Stability** | Assess backpropagation through the $\sin(\omega t)$ term. | **Mean Gradient Magnitude** of $\omega_t$ over epochs. | Smooth convergence curve; no exploding/vanishing spikes. |
| **B. Initialization Impact** | Determine if Orthogonal $\omega_t$ initialization is superior to Random initialization. | **Time-to-Convergence** (epochs needed for loss plateau). | Orthogonal init should converge faster and to a lower loss value. |
| **C. Harmonic Ablation** | Determine the required minimum number of harmonics $m$. | **Training Loss** and subsequent **Semantic Quality ($\rho$)**. | Find the minimum $m$ that achieves peak performance. |

### Phase 2: Semantic Quality Validation (Downstream Benchmarks)

This phase uses the best-performing model (from Phase 1) to validate its output on industry-standard tasks.

| Benchmark Task | Dataset | Validation Metric | Baseline Comparison |
| :--- | :--- | :--- | :--- |
| **Word Similarity** | WordSim-353, SimLex-999 | **Spearman Rank Correlation ($\rho$)** with human scores. | Standard Word2Vec/GloVe $\rho$ scores. |
| **Sentence Classification** | SST-2 (Sentiment), AG News (Topic) | **Classification Accuracy** on test set. | Averaged static Word2Vec/GloVe embeddings passed to a Logistic Regression. |
| **Contextual Test** | Use sentences where word meaning is ambiguous (e.g., "bank"). | Cosine similarity between $\mathbf{E}_{\text{river bank}}$ and $\mathbf{E}_{\text{financial bank}}$. | A successful model should show the contextual vectors are far apart, unlike static embeddings. |

## IV. Open Datasets

The recommended datasets for initial development and final large-scale validation are:

1. **Penn Treebank (PTB):** *Small, clean for fast iteration and stability testing.*

2. **Wikipedia Corpus:** *Large-scale, general domain for final semantic quality testing.*

3. **IMDb Movie Reviews:** *Good for the final Sentiment Classification benchmark.*
