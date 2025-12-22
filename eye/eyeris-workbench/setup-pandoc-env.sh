#!/bin/bash

# Source this file to use local pandoc installation
# Usage: source ./setup-pandoc-env.sh

LOCAL_PANDOC_DIR="$HOME/.local/bin"

if [ -x "$LOCAL_PANDOC_DIR/pandoc" ]; then
    export PATH="$LOCAL_PANDOC_DIR:$PATH"
    echo "Local pandoc added to PATH"
    echo "Pandoc location: $(which pandoc)"
    echo "Pandoc version: $(pandoc --version | head -1)"
else
    echo "WARNING: Local pandoc not found at $LOCAL_PANDOC_DIR/pandoc"
    echo "Run ./setup-local-pandoc.sh to install it first"
fi
