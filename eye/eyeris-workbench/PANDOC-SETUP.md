# Local Pandoc Installation Guide

## Problem
The cluster's pandoc 2.7.3 is too old and causes "File not found in resource path" errors during HTML report generation. Your Mac uses pandoc 3.8.3 which works correctly.

## Solution
Install pandoc 3.8.3 locally in your home directory without requiring root privileges.

## Installation Steps

### 1. Install Local Pandoc

On the cluster, run:

```bash
cd /oak/stanford/groups/awagner/yaams-haams/eyeris-workbench
./setup-local-pandoc.sh
```

This will:
- Download pandoc 3.8.3 from GitHub
- Install it to `~/.local/bin/pandoc`
- Verify the installation

Expected output:
```
Pandoc installed to: ~/.local/bin/pandoc
Testing installation...
pandoc 3.8.3
SUCCESS! Pandoc is ready to use.
```

### 2. Verify Installation

Check that your local pandoc is available:

```bash
~/.local/bin/pandoc --version
```

Should show: `pandoc 3.8.3`

### 3. Run Your Pipeline

The `run-eyeris` script now automatically detects and uses the local pandoc if available:

```bash
./cmd/run-eyeris "eyeris" "sub-015"
```

Check the logs to confirm it's using the local version:
```
[INFO] Using local pandoc at /home/username/.local/bin/pandoc
```

## Testing

Test pandoc before running the full pipeline:

```bash
./tests/test-pandoc.sh
```

Should show pandoc 3.8.3 in the output and all tests passing.

## Troubleshooting

### If installation fails

Check your architecture:
```bash
uname -m
```

The script supports:
- `x86_64` (amd64)
- `aarch64` / `arm64`

### To manually add pandoc to PATH

If you need to use the local pandoc in other contexts:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Or source the environment file:
```bash
source ./setup-pandoc-env.sh
```

### To reinstall

Simply run the installation script again:
```bash
./setup-local-pandoc.sh
```

It will overwrite the existing installation.

## What Changed

- **run-eyeris** (lines 88-97): Now checks for `~/.local/bin/pandoc` before loading the module
- **setup-local-pandoc.sh**: Downloads and installs pandoc 3.8.3 to `~/.local/bin`
- **setup-pandoc-env.sh**: Helper script to add local pandoc to PATH

## Why This Fixes the Issue

Pandoc 2.7.3 has known issues with resource paths when generating HTML from R Markdown. Version 3.8.3 includes:
- Better resource path resolution
- Improved template handling
- Bug fixes for data directory detection

This matches the version on your Mac where report generation works correctly.
