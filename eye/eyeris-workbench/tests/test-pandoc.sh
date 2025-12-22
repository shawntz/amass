#!/bin/bash

# Quick pandoc test wrapper for SLURM environment
# Usage: ./test-pandoc.sh

echo "Loading modules..."
module load system
module load pandoc/2.7.3
module load R

echo "Running pandoc diagnostics..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Rscript "$SCRIPT_DIR/test-pandoc.R"

echo ""
echo "Done! Check output above for errors."
