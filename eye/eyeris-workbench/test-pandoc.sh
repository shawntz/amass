#!/bin/bash

# Quick pandoc test wrapper for SLURM environment
# Usage: ./test-pandoc.sh

echo "Loading modules..."
module load system
module load pandoc/2.7.3
module load R

echo "Running pandoc diagnostics..."
Rscript ./test-pandoc.R

echo ""
echo "Done! Check output above for errors."
