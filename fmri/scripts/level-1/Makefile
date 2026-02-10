# ==============================================================================
# AMASS Surface-Based First-Level GLM Analysis
# ==============================================================================
# Makefile for running surface-based GLM analysis on native FreeSurfer surfaces
# ==============================================================================

.PHONY: help setup clean run-lsa-goal run-lsa-stim run-lssm-goal run-lssm-stim \
        run-all-lsa run-all-lssm run-all test-setup check-data \
        run-simple list-contrasts smooth smooth-check \
        viz-summary viz-design viz-histogram viz-surface viz-interactive viz-list

# Configuration
SHELL := /bin/bash
PYTHON := python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
SCRIPT := surface_1stlvl_analysis.py

# Default subject (override with: make run-lsa-goal SUB=074)
SUB ?= 073

# Default parameters
TR ?= 2.0
SLICE_TIME_REF ?= 0.5
MOTION_SCRUB ?= 0.9
MOTION_EXCLUDE ?= 5.0
DERIVATIVES ?= /Users/shawnschwartz/Developer/datasci-homelab/volumes/home/work/amass/derivatives
RUN_EXCLUDED ?=
NUISANCE ?= collapsed

# Simple GLM contrast (required for run-simple)
# Example: CONTRAST="goal_HIT > goal_MISS"
CONTRAST ?=
CONTRAST_FILE ?=

# Condition column (which column in onsets file defines trial types)
# Default: glm_goal_conds
# Example: ret_goal_recoded (for BS vs LNL)
CONDITION_COL ?= glm_goal_conds

# Smoothing parameters (for smooth command)
SMOOTH_FWHM ?= 4
SMOOTH_SPACE ?= fsnative
SMOOTH_SCRIPT := smooth_surface_data.sh

# Use pre-smoothed data for GLM (set to smoothing FWHM, e.g., SMOOTHED_FWHM=4)
SMOOTHED_FWHM ?=

# Build common arguments
COMMON_ARGS := --subID $(SUB) \
               --TR $(TR) \
               --slice_time_ref $(SLICE_TIME_REF) \
               --motion_threshold_scrub $(MOTION_SCRUB) \
               --motion_threshold_exclude $(MOTION_EXCLUDE) \
               --derivatives_root $(DERIVATIVES) \
               --nuisance_model $(NUISANCE) \
               --condition_column $(CONDITION_COL)

ifdef RUN_EXCLUDED
    COMMON_ARGS += --run_excluded "$(RUN_EXCLUDED)"
endif

ifdef SMOOTHED_FWHM
    COMMON_ARGS += --smoothed_fwhm $(SMOOTHED_FWHM)
endif

# ==============================================================================
# HELP
# ==============================================================================

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║           AMASS Surface-Based First-Level GLM Analysis                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "SETUP COMMANDS:"
	@echo "  make setup           Create virtual environment and install dependencies"
	@echo "  make test-setup      Verify installation and check data availability"
	@echo "  make check-data      Check if required data files exist for a subject"
	@echo "  make clean           Remove virtual environment and cached files"
	@echo ""
	@echo "PREPROCESSING COMMANDS:"
	@echo "  make smooth          Smooth surface data (requires wb_command)"
	@echo "  make smooth-check    Dry run - show what would be smoothed"
	@echo ""
	@echo "ANALYSIS COMMANDS:"
	@echo "  make run-simple      Run standard GLM with contrast (requires CONTRAST=...)"
	@echo "  make list-contrasts  List predefined contrasts"
	@echo "  make run-lsa-goal    Run LSA GLM for goal events (single-trial betas)"
	@echo "  make run-lsa-stim    Run LSA GLM for stim events (single-trial betas)"
	@echo "  make run-lssm-goal   Run LSSM GLM for goal events (slower, more stable)"
	@echo "  make run-lssm-stim   Run LSSM GLM for stim events (slower, more stable)"
	@echo "  make run-all-lsa     Run LSA GLM for both goal and stim events"
	@echo "  make run-all-lssm    Run LSSM GLM for both goal and stim events"
	@echo "  make run-all         Run all analyses (LSA + LSSM for goal + stim)"
	@echo ""
	@echo "VISUALIZATION COMMANDS (require PKL=path/to/results.pkl):"
	@echo "  make viz-summary     Print summary statistics"
	@echo "  make viz-design      Plot design matrices"
	@echo "  make viz-histogram   Plot beta value histogram"
	@echo "  make viz-surface     Plot beta map on surface (use TRIAL=N)"
	@echo "  make viz-interactive Open interactive surface viewer (use TRIAL=N)"
	@echo "  make viz-list        List all trials in the output file"
	@echo "  make viz-all         Run all visualizations"
	@echo ""
	@echo "CONFIGURABLE PARAMETERS (override with VAR=value):"
	@echo "  SUB                  Subject ID without 'sub-' prefix (default: $(SUB))"
	@echo "  TR                   Repetition time in seconds (default: $(TR))"
	@echo "  SLICE_TIME_REF       Slice time reference 0-1 (default: $(SLICE_TIME_REF))"
	@echo "  MOTION_SCRUB         FD threshold for scrubbing in mm (default: $(MOTION_SCRUB))"
	@echo "  MOTION_EXCLUDE       FD threshold for run exclusion in mm (default: $(MOTION_EXCLUDE))"
	@echo "  NUISANCE             Nuisance model: 'collapsed' or 'by_condition' (default: $(NUISANCE))"
	@echo "  RUN_EXCLUDED         Runs to exclude, comma-separated (default: none)"
	@echo "  DERIVATIVES          Path to derivatives directory"
	@echo ""
	@echo "SIMPLE GLM PARAMETERS:"
	@echo "  CONTRAST             Contrast string, e.g., 'goal_HIT > goal_MISS'"
	@echo "  CONTRAST_FILE        JSON file with contrast definitions"
	@echo "  CONDITION_COL        Column for trial types (default: glm_goal_conds)"
	@echo "                       Other options: ret_goal_recoded, goal_congruence, etc."
	@echo ""
	@echo "SMOOTHING PARAMETERS:"
	@echo "  SMOOTH_FWHM          Smoothing FWHM in mm for smoothing script (default: $(SMOOTH_FWHM))"
	@echo "  SMOOTH_SPACE         Surface space: fsnative or fsaverage5 (default: $(SMOOTH_SPACE))"
	@echo "  SMOOTHED_FWHM        Use pre-smoothed data for GLM (e.g., SMOOTHED_FWHM=4)"
	@echo "                       When set, uses files from derivatives/sub-{ID}/fMRI/"
	@echo ""
	@echo "VISUALIZATION PARAMETERS:"
	@echo "  PKL                  Path to .pkl results file (required for viz commands)"
	@echo "  TRIAL                Trial index for surface plots (default: $(TRIAL))"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make setup                                    # First-time setup"
	@echo "  make smooth SUB=073                           # Smooth surface data for subject"
	@echo "  make smooth SUB=073 SMOOTH_FWHM=6             # Smooth with 6mm FWHM"
	@echo "  make smooth-check SUB=073                     # Dry run (preview smoothing)"
	@echo "  make list-contrasts                           # Show predefined contrasts"
	@echo "  make run-simple CONTRAST='goal_HIT > goal_MISS'  # Simple GLM with contrast"
	@echo "  make run-simple SUB=074 CONTRAST='stim_congruent > stim_incongruent'"
	@echo "  make run-simple CONDITION_COL=ret_goal_recoded CONTRAST='goal_BS > goal_LNL'"
	@echo "  make run-simple CONTRAST='goal_HIT > goal_MISS' SMOOTHED_FWHM=4  # Use smoothed data"
	@echo "  make run-lsa-stim                             # Run LSA for stim (collapsed nuisance)"
	@echo "  make run-lsa-goal SMOOTHED_FWHM=4             # LSA with pre-smoothed 4mm data"
	@echo "  make run-lsa-goal SUB=074                     # Run LSA for goal, subject 074"
	@echo "  make run-lssm-stim NUISANCE=by_condition      # LSSM with nuisance by condition"
	@echo "  make run-lssm-stim RUN_EXCLUDED=1,3           # Run LSSM excluding runs 1 and 3"
	@echo "  make run-all SUB=073                          # Run all analyses for subject 073"
	@echo ""
	@echo "VISUALIZATION EXAMPLES:"
	@echo "  make viz-summary PKL=output.pkl               # Show summary"
	@echo "  make viz-histogram PKL=output.pkl             # Plot beta histogram"
	@echo "  make viz-surface PKL=output.pkl TRIAL=5       # Plot surface for trial 5"
	@echo "  make viz-interactive PKL=output.pkl TRIAL=0   # Interactive viewer"
	@echo ""
	@echo "DESIGN MATRIX STRUCTURE:"
	@echo "  Both goal AND stim events are always included in the model."
	@echo "  The implicit baseline is the inter-trial interval (ITI)."
	@echo ""
	@echo "  --event determines which event type gets single-trial betas extracted."
	@echo "  --nuisance_model (NUISANCE) controls how the OTHER event type is modeled:"
	@echo "    - collapsed:    Single regressor (e.g., 'goal_all' when targeting stim)"
	@echo "    - by_condition: Separate regressors (e.g., 'goal_congruent_old_HIT')"
	@echo ""
	@echo "GLM TYPE COMPARISON:"
	@echo "  Simple - Standard univariate GLM: Conditions pooled, then compute contrasts."
	@echo "           Use for traditional activation analyses (HIT > MISS, etc.)"
	@echo "  LSA    - Least Squares All: All trials in one model. Faster, but potentially"
	@echo "           unstable betas due to collinearity between adjacent trials."
	@echo "  LSSM   - Least Squares Separate Models: Each trial gets its own GLM."
	@echo "           Slower but provides more stable single-trial beta estimates."
	@echo ""
	@echo "OUTPUT:"
	@echo "  Results saved to: $(DERIVATIVES)/sub-{SUB}/fMRI/1stlvl_native_surface/"
	@echo ""

# ==============================================================================
# SETUP
# ==============================================================================

setup: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Setup complete! Virtual environment created at $(VENV_DIR)"
	@echo "Run 'make test-setup' to verify installation"
	@echo ""

$(VENV_DIR)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Upgrading pip..."
	$(VENV_PIP) install --upgrade pip
	@echo "Installing dependencies..."
	$(VENV_PIP) install -r requirements.txt
	@touch $(VENV_DIR)/bin/activate

test-setup: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Testing setup..."
	@echo "================"
	@echo ""
	@echo "Python version:"
	@$(VENV_PYTHON) --version
	@echo ""
	@echo "Checking nilearn installation..."
	@$(VENV_PYTHON) -c "import nilearn; print(f'nilearn version: {nilearn.__version__}')"
	@echo ""
	@echo "Checking numpy installation..."
	@$(VENV_PYTHON) -c "import numpy; print(f'numpy version: {numpy.__version__}')"
	@echo ""
	@echo "Checking pandas installation..."
	@$(VENV_PYTHON) -c "import pandas; print(f'pandas version: {pandas.__version__}')"
	@echo ""
	@echo "All dependencies installed correctly!"
	@echo ""
	@$(MAKE) check-data

check-data:
	@echo ""
	@echo "Checking data for subject: sub-$(SUB)"
	@echo "======================================"
	@echo ""
	@echo "Checking surface files..."
	@test -f $(DERIVATIVES)/fmriprep-24.0.1/sourcedata/freesurfer/sub-$(SUB)/surf/lh.inflated \
		&& echo "  [OK] Left hemisphere surface" \
		|| echo "  [MISSING] Left hemisphere surface"
	@test -f $(DERIVATIVES)/fmriprep-24.0.1/sourcedata/freesurfer/sub-$(SUB)/surf/rh.inflated \
		&& echo "  [OK] Right hemisphere surface" \
		|| echo "  [MISSING] Right hemisphere surface"
	@echo ""
	@echo "Checking behavioral events file..."
	@test -f $(DERIVATIVES)/fmriprep-24.0.1/sub-$(SUB)/beh/sub-$(SUB)_onsets.csv \
		&& echo "  [OK] Events file found" \
		|| echo "  [MISSING] Events file"
	@echo ""
	@echo "Checking functional data files..."
	@for run in 01 02 03 04 05 06; do \
		if test -f $(DERIVATIVES)/fmriprep-24.0.1/sub-$(SUB)/func/sub-$(SUB)_task-amass_dir-PA_run-$${run}_hemi-L_space-fsnative_bold.func.gii; then \
			echo "  [OK] Run $${run} functional data"; \
		else \
			echo "  [--] Run $${run} not found (may not exist)"; \
		fi; \
	done
	@echo ""
	@echo "Checking confounds files..."
	@for run in 01 02 03 04 05 06; do \
		if test -f $(DERIVATIVES)/fmriprep-24.0.1/sub-$(SUB)/func/sub-$(SUB)_task-amass_dir-PA_run-$${run}_desc-confounds_timeseries.tsv; then \
			echo "  [OK] Run $${run} confounds"; \
		else \
			echo "  [--] Run $${run} not found (may not exist)"; \
		fi; \
	done
	@echo ""

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete!"

# ==============================================================================
# PREPROCESSING COMMANDS
# ==============================================================================

smooth:
	@echo ""
	@echo "Smoothing surface data for sub-$(SUB)..."
	@echo "FWHM: $(SMOOTH_FWHM)mm, Space: $(SMOOTH_SPACE)"
	@echo ""
	./$(SMOOTH_SCRIPT) --subjects $(SUB) --fwhm $(SMOOTH_FWHM) --space $(SMOOTH_SPACE) \
		--derivatives $(DERIVATIVES) --fmriprep $(DERIVATIVES)/fmriprep-24.0.1

smooth-check:
	@echo ""
	@echo "Dry run: checking what would be smoothed for sub-$(SUB)..."
	@echo ""
	./$(SMOOTH_SCRIPT) --subjects $(SUB) --fwhm $(SMOOTH_FWHM) --space $(SMOOTH_SPACE) \
		--derivatives $(DERIVATIVES) --fmriprep $(DERIVATIVES)/fmriprep-24.0.1 --dry-run

# ==============================================================================
# ANALYSIS COMMANDS
# ==============================================================================

# Helper to check CONTRAST is set
check-contrast:
ifndef CONTRAST
ifndef CONTRAST_FILE
	$(error CONTRAST is not set. Usage: make run-simple CONTRAST='goal_HIT > goal_MISS')
endif
endif

list-contrasts: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Listing predefined contrasts..."
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) --list-contrasts

# Build simple GLM arguments
SIMPLE_ARGS := --subID $(SUB) \
               --TR $(TR) \
               --slice_time_ref $(SLICE_TIME_REF) \
               --motion_threshold_scrub $(MOTION_SCRUB) \
               --motion_threshold_exclude $(MOTION_EXCLUDE) \
               --derivatives_root $(DERIVATIVES) \
               --condition_column $(CONDITION_COL) \
               --GLM_type simple

ifdef CONTRAST
    SIMPLE_ARGS += --contrast "$(CONTRAST)"
endif

ifdef CONTRAST_FILE
    SIMPLE_ARGS += --contrast-file "$(CONTRAST_FILE)"
endif

ifdef RUN_EXCLUDED
    SIMPLE_ARGS += --run_excluded "$(RUN_EXCLUDED)"
endif

ifdef SMOOTHED_FWHM
    SIMPLE_ARGS += --smoothed_fwhm $(SMOOTHED_FWHM)
endif

run-simple: $(VENV_DIR)/bin/activate check-contrast
	@echo ""
	@echo "Running Simple GLM for sub-$(SUB)..."
ifdef CONTRAST
	@echo "Contrast: $(CONTRAST)"
endif
ifdef CONTRAST_FILE
	@echo "Contrast file: $(CONTRAST_FILE)"
endif
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) $(SIMPLE_ARGS)

run-lsa-goal: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Running LSA GLM for goal events (sub-$(SUB), nuisance=$(NUISANCE))..."
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) $(COMMON_ARGS) --event goal --GLM_type LSA

run-lsa-stim: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Running LSA GLM for stim events (sub-$(SUB), nuisance=$(NUISANCE))..."
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) $(COMMON_ARGS) --event stim --GLM_type LSA

run-lssm-goal: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Running LSSM GLM for goal events (sub-$(SUB), nuisance=$(NUISANCE))..."
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) $(COMMON_ARGS) --event goal --GLM_type LSSM

run-lssm-stim: $(VENV_DIR)/bin/activate
	@echo ""
	@echo "Running LSSM GLM for stim events (sub-$(SUB), nuisance=$(NUISANCE))..."
	@echo ""
	$(VENV_PYTHON) $(SCRIPT) $(COMMON_ARGS) --event stim --GLM_type LSSM

run-all-lsa: run-lsa-goal run-lsa-stim
	@echo ""
	@echo "Completed all LSA analyses for sub-$(SUB)"
	@echo ""

run-all-lssm: run-lssm-goal run-lssm-stim
	@echo ""
	@echo "Completed all LSSM analyses for sub-$(SUB)"
	@echo ""

run-all: run-all-lsa run-all-lssm
	@echo ""
	@echo "============================================"
	@echo "Completed all analyses for sub-$(SUB)"
	@echo "============================================"
	@echo ""
	@echo "Output files located at:"
	@echo "  $(DERIVATIVES)/sub-$(SUB)/fMRI/1stlvl_native_surface/"
	@echo ""

# ==============================================================================
# VISUALIZATION COMMANDS
# ==============================================================================

# Default pkl file (override with: make viz-summary PKL=path/to/file.pkl)
PKL ?=
TRIAL ?= 0
VIZ_SCRIPT := visualize_results.py

# Helper to check PKL is set
check-pkl:
ifndef PKL
	$(error PKL is not set. Usage: make viz-summary PKL=path/to/results.pkl)
endif

viz-summary: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Showing summary for: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --summary

viz-design: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Plotting design matrices for: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --design-matrices

viz-histogram: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Plotting beta histogram for: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --histogram

viz-surface: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Plotting surface for trial $(TRIAL) in: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --surface --trial $(TRIAL) --subID $(SUB) --derivatives $(DERIVATIVES)

viz-interactive: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Opening interactive viewer for trial $(TRIAL) in: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --interactive --trial $(TRIAL) --subID $(SUB) --derivatives $(DERIVATIVES)

viz-list: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Listing trials in: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --list-trials

viz-all: $(VENV_DIR)/bin/activate check-pkl
	@echo ""
	@echo "Running all visualizations for: $(PKL)"
	@echo ""
	$(VENV_PYTHON) $(VIZ_SCRIPT) $(PKL) --all
