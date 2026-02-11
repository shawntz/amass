#!/usr/bin/env bash
#
# fetch_subject_data.sh
#
# Downloads fMRIPrep derivatives, FreeSurfer surfaces, and behavioral
# event files from Sherlock for one or more AMASS subjects via SCP.
#
# Usage:
#   ./fetch_subject_data.sh 073
#   ./fetch_subject_data.sh 073 074 075
#   ./fetch_subject_data.sh --dry-run 073
#
set -euo pipefail

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
REMOTE_HOST="sherlock"
REMOTE_FMRIPREP="/oak/stanford/groups/awagner/yaams-haams/derivatives/fmriprep-24.0.1"
REMOTE_EVENTS="/oak/stanford/groups/awagner/yaams-haams/event_files"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DERIV="$(cd "${SCRIPT_DIR}/../derivatives" && pwd)"
LOCAL_FMRIPREP="${LOCAL_DERIV}/fmriprep-24.0.1"

TASK="amass"
DIR="PA"
SPACE="fsnative"
RUNS=(1 2 3 4 5 6)

DRY_RUN=false

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--runs 1,2,3] SUBJECT_ID [SUBJECT_ID ...]

Downloads fMRIPrep derivatives for the given subject(s) from Sherlock.

Options:
  --dry-run       Print the scp commands without executing them
  --runs LIST     Comma-separated run numbers (default: 1,2,3,4,5,6)
  -h, --help      Show this help message

Examples:
  $(basename "$0") 073
  $(basename "$0") --runs 1,2,3 073 074
  $(basename "$0") --dry-run 073
EOF
    exit 0
}

log()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$1"; }
warn() { printf "\033[1;33m[WARN]\033[0m  %s\n" "$1"; }
err()  { printf "\033[1;31m[ERROR]\033[0m %s\n" "$1" >&2; }

run_scp() {
    local src="$1"
    local dst="$2"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "  [DRY RUN] scp ${src} ${dst}"
    else
        if scp -q "${src}" "${dst}"; then
            log "  ✓ $(basename "${dst}")"
        else
            warn "  ✗ Failed: $(basename "${src}")"
            return 1
        fi
    fi
}

# ──────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────
SUBJECTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --runs)
            IFS=',' read -ra RUNS <<< "$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            SUBJECTS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    err "No subject IDs provided."
    usage
fi

# ──────────────────────────────────────────────
# Download files for each subject
# ──────────────────────────────────────────────
for SUB_ID in "${SUBJECTS[@]}"; do
    SUB="sub-${SUB_ID}"
    log "════════════════════════════════════════"
    log "Fetching data for ${SUB}"
    log "════════════════════════════════════════"

    FAIL_COUNT=0
    FILE_COUNT=0

    # --- Local directory setup ---
    LOCAL_ANAT="${LOCAL_FMRIPREP}/${SUB}/anat"
    LOCAL_BEH="${LOCAL_FMRIPREP}/${SUB}/beh"
    LOCAL_FUNC="${LOCAL_FMRIPREP}/${SUB}/func"
    LOCAL_SURF="${LOCAL_FMRIPREP}/sourcedata/freesurfer/${SUB}/surf"

    mkdir -p "${LOCAL_ANAT}" "${LOCAL_BEH}" "${LOCAL_FUNC}" "${LOCAL_SURF}"

    # ── 1. Anatomical surfaces (white matter) ──
    log "Downloading anatomical surfaces..."
    for HEMI in L R; do
        FNAME="${SUB}_hemi-${HEMI}_white.surf.gii"
        run_scp \
            "${REMOTE_HOST}:${REMOTE_FMRIPREP}/${SUB}/anat/${FNAME}" \
            "${LOCAL_ANAT}/${FNAME}" \
            || ((FAIL_COUNT++))
        ((FILE_COUNT++))
    done

    # ── 2. Functional data + confounds (per run) ──
    log "Downloading functional data (${#RUNS[@]} runs)..."
    for RUN_NUM in "${RUNS[@]}"; do
        RUN=$(printf "%02d" "${RUN_NUM}")

        # Confounds timeseries
        FNAME="${SUB}_task-${TASK}_dir-${DIR}_run-${RUN}_desc-confounds_timeseries.tsv"
        run_scp \
            "${REMOTE_HOST}:${REMOTE_FMRIPREP}/${SUB}/func/${FNAME}" \
            "${LOCAL_FUNC}/${FNAME}" \
            || ((FAIL_COUNT++))
        ((FILE_COUNT++))

        # GIFTI functional data (L + R hemispheres)
        for HEMI in L R; do
            FNAME="${SUB}_task-${TASK}_dir-${DIR}_run-${RUN}_hemi-${HEMI}_space-${SPACE}_bold.func.gii"
            run_scp \
                "${REMOTE_HOST}:${REMOTE_FMRIPREP}/${SUB}/func/${FNAME}" \
                "${LOCAL_FUNC}/${FNAME}" \
                || ((FAIL_COUNT++))
            ((FILE_COUNT++))
        done
    done

    # ── 3. FreeSurfer inflated surfaces ──
    log "Downloading FreeSurfer inflated surfaces..."
    for HEMI in lh rh; do
        FNAME="${HEMI}.inflated"
        run_scp \
            "${REMOTE_HOST}:${REMOTE_FMRIPREP}/sourcedata/freesurfer/${SUB}/surf/${FNAME}" \
            "${LOCAL_SURF}/${FNAME}" \
            || ((FAIL_COUNT++))
        ((FILE_COUNT++))
    done

    # ── 4. Behavioral event files ──
    log "Downloading behavioral event file..."
    FNAME="${SUB}_onsets.csv"
    run_scp \
        "${REMOTE_HOST}:${REMOTE_EVENTS}/${FNAME}" \
        "${LOCAL_BEH}/${FNAME}" \
        || ((FAIL_COUNT++))
    ((FILE_COUNT++))

    # ── Summary ──
    log "────────────────────────────────────────"
    if [[ ${FAIL_COUNT} -eq 0 ]]; then
        log "${SUB}: All ${FILE_COUNT} files downloaded successfully."
    else
        warn "${SUB}: ${FAIL_COUNT}/${FILE_COUNT} files failed."
    fi
    echo
done
