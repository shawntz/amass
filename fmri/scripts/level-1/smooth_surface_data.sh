#!/bin/bash
# ==============================================================================
# Surface Smoothing Script (Laptop Version)
# ==============================================================================
# Smooth surface-based fMRI data using Connectome Workbench
# Adapted for local execution without SLURM
# ==============================================================================

set -euo pipefail

# Default paths (adjust these to your setup)
DERIVATIVES_DIR="/Users/shawnschwartz/Developer/datasci-homelab/volumes/home/work/amass/derivatives"
FMRIPREP_DIR="${DERIVATIVES_DIR}/fmriprep-24.0.1"

# Smoothing parameters
# FWHM = 2.355 * sigma, so for 4mm FWHM, sigma = 4/2.355 = 1.699
SMOOTHING_SIGMA=1.699  # ~4mm FWHM
SMOOTHING_FWHM=4       # For display purposes

# Surface space options: "fsnative" or "fsaverage5"
SURFACE_SPACE="fsnative"

# Subjects to process (space-separated list or "all")
SUBJECTS=""

# Dry run mode
DRY_RUN=false

# ==============================================================================
# Parse command line arguments
# ==============================================================================
print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Smooth surface-based fMRI data using Connectome Workbench.

OPTIONS:
    --derivatives DIR    Path to derivatives directory
                         (default: $DERIVATIVES_DIR)
    --fmriprep DIR       Path to fMRIPrep output directory
                         (default: $FMRIPREP_DIR)
    --subjects LIST      Space-separated list of subject IDs (without 'sub-')
                         or "all" to process all subjects
    --space SPACE        Surface space: 'fsnative' or 'fsaverage5'
                         (default: $SURFACE_SPACE)
    --sigma VALUE        Smoothing sigma in mm (default: $SMOOTHING_SIGMA)
    --fwhm VALUE         Smoothing FWHM in mm (overrides sigma)
    --dry-run            Show what would be done without executing
    -h, --help           Show this help message

EXAMPLES:
    # Smooth a single subject in native space
    $(basename "$0") --subjects 073

    # Smooth multiple subjects
    $(basename "$0") --subjects "073 074 075"

    # Smooth all subjects in fsaverage5 space with 6mm FWHM
    $(basename "$0") --subjects all --space fsaverage5 --fwhm 6

    # Dry run to see what would be processed
    $(basename "$0") --subjects 073 --dry-run

NOTES:
    - Requires Connectome Workbench (wb_command) to be installed
    - For fsnative space, uses FreeSurfer white surface from fMRIPrep
    - For fsaverage5 space, requires fsaverage5 surfaces in derivatives
    - Output files are saved to: derivatives/sub-{ID}/fMRI/

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --derivatives)
            DERIVATIVES_DIR="$2"
            shift 2
            ;;
        --fmriprep)
            FMRIPREP_DIR="$2"
            shift 2
            ;;
        --subjects)
            SUBJECTS="$2"
            shift 2
            ;;
        --space)
            SURFACE_SPACE="$2"
            shift 2
            ;;
        --sigma)
            SMOOTHING_SIGMA="$2"
            shift 2
            ;;
        --fwhm)
            SMOOTHING_FWHM="$2"
            # Convert FWHM to sigma: sigma = FWHM / 2.355
            SMOOTHING_SIGMA=$(echo "scale=4; $2 / 2.355" | bc)
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# Validation
# ==============================================================================
echo ""
echo "============================================================"
echo "Surface Smoothing Script"
echo "============================================================"
echo ""

# Check for wb_command (also check common installation locations)
WB_COMMAND=""
if command -v wb_command &> /dev/null; then
    WB_COMMAND=$(which wb_command)
elif [[ -f "/Applications/workbench/bin_macosx64/wb_command" ]]; then
    WB_COMMAND="/Applications/workbench/bin_macosx64/wb_command"
elif [[ -f "/Applications/workbench/bin_macosxub/wb_command" ]]; then
    WB_COMMAND="/Applications/workbench/bin_macosxub/wb_command"
elif [[ -f "/usr/local/bin/wb_command" ]]; then
    WB_COMMAND="/usr/local/bin/wb_command"
fi

if [[ -z "$WB_COMMAND" ]]; then
    echo "[ERROR] wb_command not found. Please install Connectome Workbench."
    echo "        macOS: brew install --cask connectome-workbench"
    echo "        Or download from: https://www.humanconnectome.org/software/get-connectome-workbench"
    exit 1
fi
echo "[OK] wb_command found: $WB_COMMAND"

# Validate directories
if [[ ! -d "$DERIVATIVES_DIR" ]]; then
    echo "[ERROR] Derivatives directory not found: $DERIVATIVES_DIR"
    exit 1
fi
echo "[OK] Derivatives directory: $DERIVATIVES_DIR"

if [[ ! -d "$FMRIPREP_DIR" ]]; then
    echo "[ERROR] fMRIPrep directory not found: $FMRIPREP_DIR"
    exit 1
fi
echo "[OK] fMRIPrep directory: $FMRIPREP_DIR"

# Validate surface space
if [[ "$SURFACE_SPACE" != "fsnative" && "$SURFACE_SPACE" != "fsaverage5" ]]; then
    echo "[ERROR] Invalid surface space: $SURFACE_SPACE (must be 'fsnative' or 'fsaverage5')"
    exit 1
fi
echo "[OK] Surface space: $SURFACE_SPACE"

# Check subjects
if [[ -z "$SUBJECTS" ]]; then
    echo "[ERROR] No subjects specified. Use --subjects"
    print_usage
    exit 1
fi

# Build subject list
if [[ "$SUBJECTS" == "all" ]]; then
    SUBJECT_LIST=()
    for subdir in "$FMRIPREP_DIR"/sub-*/; do
        if [[ -d "$subdir" ]]; then
            subid=$(basename "$subdir" | sed 's/sub-//')
            SUBJECT_LIST+=("$subid")
        fi
    done
else
    read -ra SUBJECT_LIST <<< "$SUBJECTS"
fi

echo "[OK] Subjects to process: ${SUBJECT_LIST[*]}"
echo "[OK] Smoothing: ${SMOOTHING_FWHM}mm FWHM (sigma=${SMOOTHING_SIGMA}mm)"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[DRY RUN MODE - No files will be modified]"
fi

echo ""

# ==============================================================================
# Get surface file path
# ==============================================================================
get_surface_file() {
    local subid="$1"
    local hemi="$2"  # L or R

    if [[ "$SURFACE_SPACE" == "fsnative" ]]; then
        # First try: GIFTI surface in anat folder (fMRIPrep output)
        local anat_dir="$FMRIPREP_DIR/sub-${subid}/anat"
        local gii_file="${anat_dir}/sub-${subid}_hemi-${hemi}_white.surf.gii"

        if [[ -f "$gii_file" ]]; then
            echo "$gii_file"
            return
        fi

        # Second try: GIFTI in FreeSurfer surf folder
        local fs_dir="$FMRIPREP_DIR/sourcedata/freesurfer/sub-${subid}/surf"
        local hemi_lower=$(echo "$hemi" | tr '[:upper:]' '[:lower:]')
        local fs_gii_file="${fs_dir}/${hemi_lower}h.white.surf.gii"

        if [[ -f "$fs_gii_file" ]]; then
            echo "$fs_gii_file"
            return
        fi

        # Third try: FreeSurfer binary format (will fail with wb_command but we check anyway)
        local fs_binary="${fs_dir}/${hemi_lower}h.white"
        if [[ -f "$fs_binary" ]]; then
            echo "$fs_binary"
            return
        fi

        echo ""
    else
        # fsaverage5 surfaces should be in derivatives
        local surf_file="$DERIVATIVES_DIR/fsaverage5_hemi-${hemi}.white.surf.gii"
        if [[ -f "$surf_file" ]]; then
            echo "$surf_file"
        else
            # Try alternative naming
            local alt_file="$DERIVATIVES_DIR/fsaverage5.hemi-${hemi}.white.surf.gii"
            if [[ -f "$alt_file" ]]; then
                echo "$alt_file"
            else
                echo ""
            fi
        fi
    fi
}

# ==============================================================================
# Process each subject
# ==============================================================================
total_subjects=${#SUBJECT_LIST[@]}
current_subject=0
total_smoothed=0
total_skipped=0
total_errors=0

for subid in "${SUBJECT_LIST[@]}"; do
    ((current_subject++))
    echo "============================================================"
    echo "[$current_subject/$total_subjects] Processing sub-${subid}"
    echo "============================================================"

    subdir="$FMRIPREP_DIR/sub-${subid}"

    if [[ ! -d "$subdir" ]]; then
        echo "[WARN] Subject directory not found: $subdir"
        echo "[SKIP] Skipping sub-${subid}"
        ((total_skipped++))
        continue
    fi

    # Find functional files based on surface space
    if [[ "$SURFACE_SPACE" == "fsnative" ]]; then
        pattern="$subdir/func/*_space-fsnative_bold.func.gii"
    else
        pattern="$subdir/func/*_space-fsaverage5_bold.func.gii"
    fi

    func_files=($pattern)

    if [[ ${#func_files[@]} -eq 0 || ! -f "${func_files[0]}" ]]; then
        echo "[WARN] No ${SURFACE_SPACE} func.gii files found for sub-${subid}"
        echo "[SKIP] Skipping sub-${subid}"
        ((total_skipped++))
        continue
    fi

    echo "[INFO] Found ${#func_files[@]} functional files to smooth"

    # Create output directory
    output_dir="$DERIVATIVES_DIR/sub-${subid}/fMRI"
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$output_dir"
    fi

    # Process each functional file
    for func_file in "${func_files[@]}"; do
        filename=$(basename "$func_file")
        echo ""
        echo "  Processing: $filename"

        # Parse hemisphere from filename
        if [[ "$filename" =~ hemi-([LR]) ]]; then
            hemi="${BASH_REMATCH[1]}"
        else
            echo "  [ERROR] Could not parse hemisphere from: $filename"
            ((total_errors++))
            continue
        fi

        # Get surface file
        surface_file=$(get_surface_file "$subid" "$hemi")

        if [[ -z "$surface_file" || ! -f "$surface_file" ]]; then
            echo "  [ERROR] Surface file not found for hemisphere $hemi"
            echo "         Expected: FreeSurfer surface in $FMRIPREP_DIR/sourcedata/freesurfer/sub-${subid}/surf/"
            ((total_errors++))
            continue
        fi

        echo "  Surface: $(basename "$surface_file")"

        # Create output filename
        output_file="$output_dir/${filename%.func.gii}_smoothed${SMOOTHING_FWHM}mm.func.gii"

        # Check if output already exists
        if [[ -f "$output_file" ]]; then
            echo "  [SKIP] Output already exists: $(basename "$output_file")"
            ((total_skipped++))
            continue
        fi

        echo "  Output: $(basename "$output_file")"

        # Run smoothing
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [DRY RUN] Would run: $WB_COMMAND -metric-smoothing \"$surface_file\" \"$func_file\" $SMOOTHING_SIGMA \"$output_file\""
        else
            echo "  Smoothing with sigma=${SMOOTHING_SIGMA}mm (FWHM=${SMOOTHING_FWHM}mm)..."

            if "$WB_COMMAND" -metric-smoothing "$surface_file" "$func_file" "$SMOOTHING_SIGMA" "$output_file"; then
                echo "  [OK] Smoothing complete"
                ((total_smoothed++))
            else
                echo "  [ERROR] Smoothing failed"
                ((total_errors++))
            fi
        fi
    done

    echo ""
done

# ==============================================================================
# Summary
# ==============================================================================
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "  Subjects processed: $total_subjects"
echo "  Files smoothed:     $total_smoothed"
echo "  Files skipped:      $total_skipped"
echo "  Errors:             $total_errors"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] No files were modified. Remove --dry-run to execute."
fi

echo "Done!"
