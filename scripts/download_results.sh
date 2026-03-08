#!/bin/bash
# Download trained results from Vast.ai instance
#
# Usage: bash scripts/download_results.sh <HOST> <PORT>
# Example: bash scripts/download_results.sh ssh5.vast.ai 12345

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <HOST> <PORT>"
    echo "Example: $0 ssh5.vast.ai 12345"
    echo ""
    echo "Find your instance details with: vastai show instances"
    exit 1
fi

HOST="$1"
PORT="$2"
REMOTE_DIR="/root/wave_embeddings/checkpoints"
LOCAL_DIR="checkpoints"

mkdir -p "$LOCAL_DIR"

echo "Downloading from ${HOST}:${PORT}..."

# Download CSV lookup table
scp -P "$PORT" "root@${HOST}:${REMOTE_DIR}/wave_v5_params.csv" "${LOCAL_DIR}/"
echo "  ✓ wave_v5_params.csv"

# Download checkpoint (optional, larger file)
scp -P "$PORT" "root@${HOST}:${REMOTE_DIR}/wave_lm_v5.pt" "${LOCAL_DIR}/" 2>/dev/null && \
    echo "  ✓ wave_lm_v5.pt" || \
    echo "  ⚠ wave_lm_v5.pt not found (CSV is the main output)"

echo ""
echo "Done! Results in ${LOCAL_DIR}/"
echo "  CSV: ${LOCAL_DIR}/wave_v5_params.csv"
