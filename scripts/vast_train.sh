#!/bin/bash
# Vast.ai one-command training setup for WaveLM v5
# Usage: bash scripts/vast_train.sh
#
# Recommended instances:
#   V100 16GB  ~$0.09/hr  — good for vocab_size <= 10K
#   A100 40GB  ~$0.29/hr  — good for vocab_size <= 30K
#
# Expected time: ~20 epochs on WikiText-2 with V=10K takes ~30-60 min on V100

set -euo pipefail

echo "=== WaveLM v5 GPU Training Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

# Install dependencies
echo "Installing dependencies..."
pip install --quiet torch datasets scipy

# Run training
echo "Starting training..."
python experiments/train_lm_v5.py \
    --device cuda \
    --epochs 20 \
    --vocab-size 10000 \
    --batch-size 16 \
    --chunk-size 4

echo ""
echo "=== Training complete ==="
echo "Outputs:"
echo "  Checkpoint: checkpoints/wave_lm_v5.pt"
echo "  CSV table:  checkpoints/wave_v5_params.csv"
echo ""
echo "To download results, run from your local machine:"
echo "  scp -P <PORT> root@<HOST>:$(pwd)/checkpoints/wave_v5_params.csv ."
