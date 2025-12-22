#!/bin/bash

# Setup local pandoc installation without root privileges
# This downloads and installs pandoc to a local directory

set -e

PANDOC_VERSION="3.8.3"
INSTALL_DIR="$HOME/.local"
BIN_DIR="$INSTALL_DIR/bin"
TEMP_DIR="/tmp/pandoc-install-$$"

echo "================================================"
echo "Local Pandoc Installer (No Root Required)"
echo "================================================"
echo ""
echo "Version: $PANDOC_VERSION"
echo "Install location: $INSTALL_DIR"
echo ""

# Create directories
mkdir -p "$BIN_DIR"
mkdir -p "$TEMP_DIR"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH_NAME="amd64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ARCH_NAME="arm64"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Detected architecture: $ARCH ($ARCH_NAME)"
echo ""

# Download URL
DOWNLOAD_URL="https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-${ARCH_NAME}.tar.gz"

echo "Downloading pandoc ${PANDOC_VERSION}..."
echo "URL: $DOWNLOAD_URL"
echo ""

cd "$TEMP_DIR"

# Download pandoc
if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$DOWNLOAD_URL" -O pandoc.tar.gz
elif command -v curl >/dev/null 2>&1; then
    curl -L --progress-bar "$DOWNLOAD_URL" -o pandoc.tar.gz
else
    echo "ERROR: Neither wget nor curl is available"
    exit 1
fi

echo ""
echo "Extracting archive..."
tar -xzf pandoc.tar.gz

echo "Installing to $BIN_DIR..."
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "pandoc-*" | head -1)
cp "$EXTRACTED_DIR/bin/pandoc" "$BIN_DIR/"
chmod +x "$BIN_DIR/pandoc"

# Clean up
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Pandoc installed to: $BIN_DIR/pandoc"
echo ""

# Test installation
if [ -x "$BIN_DIR/pandoc" ]; then
    echo "Testing installation..."
    "$BIN_DIR/pandoc" --version
    echo ""
    echo "SUCCESS! Pandoc is ready to use."
    echo ""
    echo "To use this pandoc, make sure $BIN_DIR is in your PATH:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
    echo "Or source the provided environment file:"
    echo "  source $(dirname "$0")/setup-pandoc-env.sh"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
