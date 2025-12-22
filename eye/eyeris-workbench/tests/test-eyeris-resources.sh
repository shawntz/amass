#!/bin/bash

# Test eyeris package resource files
# Usage: ./test-eyeris-resources.sh

echo "Loading R module..."
module load R

echo "Running eyeris resource diagnostics..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Rscript "$SCRIPT_DIR/test-eyeris-resources.R"

echo ""
echo "Done! If resources are missing, eyeris package needs to be reinstalled."
