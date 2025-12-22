SHELL := /bin/bash
SCRIPT := ./cmd/bidsify
RUNNER := ./cmd/run-eyeris
SUBMITTER := ./submit-eyeris.sbatch

.PHONY: eyeris
eyeris:
	@echo "Usage: make eyeris <subcommand>"
	@echo ""
	@if [ -z "$(start)" ] || [ -z "$(end)" ]; then \
		echo "Usage: make eyeris start=<start> end=<end>"; \
		exit 1; \
	fi; \
	echo "Running eyeris glassbox/epoch/bidsify pipeline from subject $$start to $$end" && \
	./submit-eyeris.sbatch "$$start" "$$end" "$${session}"
	echo "Done!"

.PHONY: setup
setup:
	@echo "Setting up preprocessing environment..."
	chmod +x $(SCRIPT)
	chmod +x $(RUNNER)
	chmod +x $(SUBMITTER)
	@echo "Setup complete!"

.PHONY: bidsify
bidsify: setup
	@echo "Running complete preprocessing pipeline..."
	$(SCRIPT) all

.PHONY: edf2asc
edf2asc: setup
	@echo "Converting EDF files to ASC format..."
	$(SCRIPT) edf2asc

.PHONY: move_asc
move_asc: setup
	@echo "Moving ASC files to session directories..."
	$(SCRIPT) move_asc

.PHONY: bids
bids: setup
	@echo "Organizing files in BIDS format..."
	$(SCRIPT) bids

.PHONY: stack-beh
stack-beh:
	@echo "Stacking behavioral data from ses-enc and ses-ret sessions..."
	@chmod +x ./utils/R/stack-beh-files.R
	@./utils/R/stack-beh-files.R $${input:-/Volumes/LumpySpace/prism} $${output:-./data/beh}

.PHONY: stack-gradcpt
stack-gradcpt:
	@echo "Stacking gradCPT behavioral data..."
	@chmod +x ./utils/R/stack-gradcpt-files.R
	@./utils/R/stack-gradcpt-files.R $${input:-/Volumes/LumpySpace/prism} $${output:-./data/beh}

.PHONY: merge-eyeris
merge-eyeris:
	@echo "Merging run-level eyeris RDS files into subject-level RDS files..."
	@chmod +x ./utils/R/merge-eyeris-runs.R
	@./utils/R/merge-eyeris-runs.R $${input:-/Volumes/LumpySpace/clamp/derivatives} $${output:-./data/processed/pupil/merged}

.PHONY: gradcpt
gradcpt: setup
	@echo "Organizing gradcpt files in BIDS format..."
	$(SCRIPT) gradcpt

.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	@echo "Removing ASC files..."
	find sourcedata/pupil -name "*.asc" -type f -delete
	@echo "Removing BIDS directories..."
	rm -rf sub-*
	@echo "Cleanup complete!"

.PHONY: help
help:
	@echo "AMASS Retrieval Arm Pupil Preproc Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup      - Make preprocessing script executable"
	@echo "  bidsify    - Run complete preprocessing pipeline (default)"
	@echo "  edf2asc    - Convert EDF files to ASC format only"
	@echo "  move_asc   - Move ASC files to session directories only"
	@echo "  bids       - Organize files in BIDS format only"
	@echo "  gradcpt    - Organize gradcpt files in BIDS format only"
	@echo "  stack-beh  - Stack behavioral data from ses-enc and ses-ret sessions"
	@echo "  stack-gradcpt - Stack gradCPT behavioral data from .mat files"
	@echo "  merge-eyeris - Merge run-level eyeris RDS files into subject-level RDS files"
	@echo "  clean      - Remove generated files (ASC files and BIDS directories)"
	@echo "  subjects   - Show list of currently available subjects"
	@echo "  stats      - Show file statistics"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Run all preprocessing"
	@echo "  make edf2asc      # Convert EDF files only"
	@echo "  make bids         # Organize BIDS files only"
	@echo "  make stack-beh    # Stack behavioral data (uses current directory)"
	@echo "  make stack-beh input=data output=data/merged  # Stack with custom directories"
	@echo "  make stack-gradcpt # Stack gradCPT data"
	@echo "  make stack-gradcpt input=/custom/path  # Stack with custom input directory"
	@echo "  make merge-eyeris # Merge eyeris run-level RDS files"
	@echo "  make merge-eyeris input=/custom/derivatives output=/custom/output  # Merge with custom paths"
	@echo "  make clean        # Clean up generated files"

.PHONY: check
check:
	@echo "Checking project structure..."
	@if [ ! -d "../sourcedata/eye" ]; then \
		echo "ERROR: sourcedata/eye directory not found"; \
		exit 1; \
	fi
	@if [ ! -f "./cmd/edf2asc" ]; then \
		echo "WARNING: EDF2ASC converter not found at ./cmd/edf2asc"; \
		echo "EDF conversion will fail if this is missing"; \
	fi
	@echo "Project structure check complete!"


# human-readable version
.PHONY: subjects
subjects:
	@echo "Available subjects:"
	@for dir in ../sourcedata/eye/s*; do \
		if [ -d "$$dir" ]; then \
			subject=$$(basename "$$dir"); \
			echo "  $$subject"; \
		fi; \
	done

# machine-readable version
subjects-list:
	@for dir in ../sourcedata/eye/s*; do \
		if [ -d "$$dir" ]; then \
			basename "$$dir"; \
		fi; \
	done

.PHONY: stats
stats:
	@echo "File statistics:"
	@echo "EDF files: $$(find ../sourcedata/eye -name "*.EDF" | wc -l)"
	@echo "ASC files: $$(find ../sourcedata/eye -name "*.asc" | wc -l)"
	@echo "CSV files: $$(find ../sourcedata/eye -name "*.csv" | wc -l)"
	@echo "Subject directories: $$(find ../sourcedata/eye -maxdepth 1 -type d -name "s*" | wc -l)"