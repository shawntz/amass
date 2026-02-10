#!/usr/bin/env python3
"""
Surface-based first-level GLM analysis using nilearn.

Supports three GLM approaches:
- Simple: Standard GLM with conditions pooled (for univariate contrasts)
- LSA: Least Squares All (single-trial betas, all in one model)
- LSSM: Least Squares Separate Models (single-trial betas, separate models)

Adapted for AMASS project file structure.
"""

import argparse
import glob
import json
import os
import pickle as pkl
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.surface import PolyData, PolyMesh, SurfaceImage


# =============================================================================
# GLM PARAMETERS
# =============================================================================
def get_glm_params(subID: str, TR: float, slice_time_ref: float) -> dict:
    """Return GLM parameters dictionary."""
    return {
        'drift_model': 'cosine',
        'drift_order': 1,
        'fir_delays': None,
        'high_pass': 0.01,
        'hrf_model': 'spm',
        'mask_img': None,
        'memory': None,
        'memory_level': 1,
        'min_onset': -24,
        'minimize_memory': False,
        'n_jobs': 2,
        'noise_model': 'ar1',
        'random_state': None,
        'signal_scaling': False,
        'slice_time_ref': slice_time_ref,
        'smoothing_fwhm': None,
        'standardize': False,
        'subject_label': subID,
        't_r': TR,
        'target_affine': None,
        'target_shape': None,
        'verbose': 0,
    }


# =============================================================================
# FILE PATH HELPERS
# =============================================================================
def get_surface_paths(derivatives_root: str, subID: str) -> tuple[str, str]:
    """Get left and right hemisphere inflated surface paths."""
    freesurfer_surf = os.path.join(
        derivatives_root, 'fmriprep-24.0.1', 'sourcedata', 'freesurfer',
        f'sub-{subID}', 'surf'
    )
    left_surf = os.path.join(freesurfer_surf, 'lh.inflated')
    right_surf = os.path.join(freesurfer_surf, 'rh.inflated')
    return left_surf, right_surf


def get_functional_paths(
    derivatives_root: str,
    subID: str,
    run: int,
    smoothed_fwhm: int = None,
) -> tuple[str, str]:
    """Get left and right hemisphere functional data paths for a run.

    Args:
        derivatives_root: Path to derivatives directory
        subID: Subject ID
        run: Run number
        smoothed_fwhm: If specified, use smoothed data with this FWHM (e.g., 4 for 4mm)
    """
    if smoothed_fwhm:
        # Use pre-smoothed data from derivatives/sub-{subID}/fMRI/
        func_dir = os.path.join(derivatives_root, f'sub-{subID}', 'fMRI')
        left_data = os.path.join(
            func_dir,
            f'sub-{subID}_task-amass_dir-PA_run-{run:02d}_hemi-L_space-fsnative_bold_smoothed{smoothed_fwhm}mm.func.gii'
        )
        right_data = os.path.join(
            func_dir,
            f'sub-{subID}_task-amass_dir-PA_run-{run:02d}_hemi-R_space-fsnative_bold_smoothed{smoothed_fwhm}mm.func.gii'
        )
    else:
        # Use unsmoothed data from fmriprep output
        func_dir = os.path.join(
            derivatives_root, 'fmriprep-24.0.1', f'sub-{subID}', 'func'
        )
        left_data = os.path.join(
            func_dir,
            f'sub-{subID}_task-amass_dir-PA_run-{run:02d}_hemi-L_space-fsnative_bold.func.gii'
        )
        right_data = os.path.join(
            func_dir,
            f'sub-{subID}_task-amass_dir-PA_run-{run:02d}_hemi-R_space-fsnative_bold.func.gii'
        )
    return left_data, right_data


def get_confounds_path(derivatives_root: str, subID: str, run: int) -> str:
    """Get confounds file path for a run."""
    return os.path.join(
        derivatives_root, 'fmriprep-24.0.1', f'sub-{subID}', 'func',
        f'sub-{subID}_task-amass_dir-PA_run-{run:02d}_desc-confounds_timeseries.tsv'
    )


def get_events_path(derivatives_root: str, subID: str) -> str:
    """Get behavioral events file path."""
    return os.path.join(
        derivatives_root, 'fmriprep-24.0.1', f'sub-{subID}', 'beh',
        f'sub-{subID}_onsets.csv'
    )


def get_output_dir(derivatives_root: str, subID: str) -> str:
    """Get base output directory for first-level results."""
    return os.path.join(
        derivatives_root, f'sub-{subID}', 'fMRI', '1stlvl_native_surface'
    )


def create_timestamped_output_dir(
    derivatives_root: str,
    subID: str,
    glm_type: str,
    contrast_name: str = None,
) -> str:
    """Create timestamped output directory for this analysis run."""
    base_dir = get_output_dir(derivatives_root, subID)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if contrast_name:
        # Sanitize contrast name for filesystem
        safe_name = re.sub(r'[^\w\-]', '_', contrast_name)[:50]
        dir_name = f'{glm_type}_{safe_name}_{timestamp}'
    else:
        dir_name = f'{glm_type}_{timestamp}'

    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def detect_runs(derivatives_root: str, subID: str) -> list[int]:
    """Detect available runs for a subject."""
    pattern = os.path.join(
        derivatives_root, 'fmriprep-24.0.1', f'sub-{subID}', 'func',
        f'sub-{subID}_task-amass_dir-PA_run-*_hemi-L_space-fsnative_bold.func.gii'
    )
    files = glob.glob(pattern)
    runs = []
    for f in files:
        basename = os.path.basename(f)
        for part in basename.split('_'):
            if part.startswith('run-'):
                try:
                    run_num = int(part.replace('run-', ''))
                    runs.append(run_num)
                except ValueError:
                    continue
    return sorted(set(runs))


# =============================================================================
# CONFOUNDS PROCESSING
# =============================================================================
def load_and_filter_confounds(
    confounds_path: str,
    motion_threshold_scrub: float,
    motion_threshold_exclude: float,
) -> tuple[pd.DataFrame | None, str | None, dict | None]:
    """
    Load confounds and apply motion scrubbing.

    Returns:
        filtered_confounds: DataFrame with confounds and motion scrub columns,
                           or None if run should be excluded
        skip_reason: Reason for skipping, or None if run is valid
        motion_info: Dict with motion statistics
    """
    confounds = pd.read_csv(confounds_path, sep='\t')

    filtered_confounds = confounds[[
        "framewise_displacement", "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ]].copy()

    filtered_confounds.loc[0, "framewise_displacement"] = 0

    fd = filtered_confounds["framewise_displacement"]
    max_fd = fd.max()
    mean_fd = fd.mean()
    total_frames = len(filtered_confounds)

    if max_fd > motion_threshold_exclude:
        return None, f"Max FD ({max_fd:.2f} mm) > threshold ({motion_threshold_exclude} mm)", None

    motion_frames = fd > motion_threshold_scrub
    scrubbed_trs = []

    if motion_frames.any():
        num_motion_frames = motion_frames.sum()

        if num_motion_frames / total_frames > 0.25:
            return None, f">25% TRs ({num_motion_frames}/{total_frames}) exceed motion threshold", None

        for tr_index in motion_frames[motion_frames].index:
            filtered_confounds[f"motion_TR_{tr_index}"] = (
                filtered_confounds.index == tr_index
            ).astype(int)
            scrubbed_trs.append(tr_index)

    motion_info = {
        'total_trs': total_frames,
        'max_fd': max_fd,
        'mean_fd': mean_fd,
        'scrubbed_trs': scrubbed_trs,
        'n_scrubbed': len(scrubbed_trs),
        'pct_scrubbed': 100 * len(scrubbed_trs) / total_frames,
    }

    return filtered_confounds, None, motion_info


# =============================================================================
# EVENT PROCESSING
# =============================================================================
def load_events(events_path: str) -> pd.DataFrame:
    """Load the behavioral events CSV file."""
    return pd.read_csv(events_path)


def process_events_simple(
    events: pd.DataFrame,
    run: int,
    condition_column: str = 'glm_goal_conds',
) -> pd.DataFrame:
    """
    Process events for simple GLM - both goal and stim with conditions pooled.

    Args:
        events: Full events DataFrame
        run: Run number (1-indexed)
        condition_column: Column name to use for trial type conditions

    Returns:
        Events DataFrame with goal and stim events, conditions as trial_type
    """
    run_events = events[events["run"] == run].copy()

    dummy_scan_duration = 0.0
    if "dummy_scan_end" in run_events.columns:
        dummy_scan_duration = run_events["dummy_scan_end"].iloc[0]

    if 'non_interest' in run_events.columns:
        run_events = run_events[run_events['non_interest'] == 0]

    # Create goal events
    goal_events = pd.DataFrame({
        'onset': run_events['goal_onset'] - dummy_scan_duration,
        'duration': run_events['goal_duration'],
        'trial_type': run_events[condition_column].apply(lambda x: f"goal_{x}"),
    })

    # Create stim events
    stim_events = pd.DataFrame({
        'onset': run_events['stim_onset'] - dummy_scan_duration,
        'duration': run_events['stim_duration'],
        'trial_type': run_events[condition_column].apply(lambda x: f"stim_{x}"),
    })

    # Combine and sort
    combined = pd.concat([goal_events, stim_events], ignore_index=True)
    combined = combined.sort_values(by='onset').reset_index(drop=True)

    return combined


def process_events_for_run(
    events: pd.DataFrame,
    run: int,
    target_event: str,
    nuisance_model: str = 'collapsed',
    condition_column: str = 'glm_goal_conds',
) -> pd.DataFrame:
    """
    Filter and process events for LSA/LSSM with target and nuisance separation.

    Args:
        events: Full events DataFrame
        run: Run number (1-indexed)
        target_event: 'goal' or 'stim'
        nuisance_model: 'collapsed' or 'by_condition'
        condition_column: Column name to use for trial type conditions
    """
    run_events = events[events["run"] == run].copy()

    dummy_scan_duration = 0.0
    if "dummy_scan_end" in run_events.columns:
        dummy_scan_duration = run_events["dummy_scan_end"].iloc[0]

    if 'non_interest' in run_events.columns:
        run_events = run_events[run_events['non_interest'] == 0]

    goal_events = pd.DataFrame({
        'onset': run_events['goal_onset'] - dummy_scan_duration,
        'duration': run_events['goal_duration'],
        'trial_type': run_events[condition_column].apply(lambda x: f"goal_{x}"),
        'original_condition': run_events[condition_column],
        'event_type': 'goal',
    })

    stim_events = pd.DataFrame({
        'onset': run_events['stim_onset'] - dummy_scan_duration,
        'duration': run_events['stim_duration'],
        'trial_type': run_events[condition_column].apply(lambda x: f"stim_{x}"),
        'original_condition': run_events[condition_column],
        'event_type': 'stim',
    })

    if target_event == 'goal':
        target_df = goal_events.copy()
        nuisance_df = stim_events.copy()
        nuisance_prefix = 'stim'
    else:
        target_df = stim_events.copy()
        nuisance_df = goal_events.copy()
        nuisance_prefix = 'goal'

    if nuisance_model == 'collapsed':
        nuisance_df['trial_type'] = f'{nuisance_prefix}_all'

    combined = pd.concat([target_df, nuisance_df], ignore_index=True)
    combined = combined.sort_values(by='onset').reset_index(drop=True)

    return combined


def get_target_event_count(events_df: pd.DataFrame, target_event: str) -> int:
    """Return count of target events."""
    return (events_df['event_type'] == target_event).sum()


# =============================================================================
# CONTRAST DEFINITION AND PARSING
# =============================================================================
def parse_contrast_string(contrast_str: str) -> tuple[str, dict, dict]:
    """
    Parse a contrast string into positive and negative condition weights.

    Supported formats:
        "HIT > MISS"           -> HIT:1, MISS:-1
        "HIT - MISS"           -> HIT:1, MISS:-1
        "HIT + FA > MISS + CR" -> HIT:1, FA:1, MISS:-1, CR:-1
        "2*HIT > MISS + CR"    -> HIT:2, MISS:-1, CR:-1

    Condition names use partial matching (case-insensitive).

    Returns:
        contrast_name: Sanitized name for the contrast
        positive_weights: dict of condition_pattern -> weight
        negative_weights: dict of condition_pattern -> weight
    """
    contrast_str = contrast_str.strip()

    # Determine separator
    if ' > ' in contrast_str:
        parts = contrast_str.split(' > ')
        contrast_name = contrast_str.replace(' > ', '_gt_').replace(' ', '_')
    elif ' - ' in contrast_str:
        parts = contrast_str.split(' - ')
        contrast_name = contrast_str.replace(' - ', '_minus_').replace(' ', '_')
    elif ' < ' in contrast_str:
        parts = contrast_str.split(' < ')
        parts = [parts[1], parts[0]]  # Flip for less-than
        contrast_name = contrast_str.replace(' < ', '_lt_').replace(' ', '_')
    else:
        # Single condition (vs baseline)
        parts = [contrast_str, '']
        contrast_name = contrast_str.replace(' ', '_')

    def parse_side(side_str: str) -> dict:
        """Parse one side of the contrast into condition:weight dict."""
        weights = {}
        if not side_str.strip():
            return weights

        # Split by + (keeping track of terms)
        terms = re.split(r'\s*\+\s*', side_str.strip())

        for term in terms:
            term = term.strip()
            if not term:
                continue

            # Check for weight prefix (e.g., "2*HIT" or "0.5*MISS")
            match = re.match(r'^([\d.]+)\s*\*\s*(.+)$', term)
            if match:
                weight = float(match.group(1))
                condition = match.group(2).strip()
            else:
                weight = 1.0
                condition = term

            weights[condition] = weight

        return weights

    positive_weights = parse_side(parts[0])
    negative_weights = parse_side(parts[1]) if len(parts) > 1 else {}

    return contrast_name, positive_weights, negative_weights


def load_contrast_file(filepath: str) -> list[dict]:
    """
    Load contrast definitions from a JSON file.

    Expected format:
    {
        "contrasts": [
            {
                "name": "goal_hit_vs_miss",
                "definition": "goal_congruent_old_HIT + goal_incongruent_old_HIT > goal_congruent_old_MISS + goal_incongruent_old_MISS"
            },
            ...
        ]
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('contrasts', [])


def build_contrast_vector(
    design_columns: list[str],
    positive_weights: dict,
    negative_weights: dict,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Build a contrast vector from condition patterns and weights.

    Uses partial matching: pattern "HIT" matches "goal_congruent_old_HIT".
    Weights are normalized so positive and negative sides sum to +1 and -1.

    Returns:
        contrast: The contrast vector
        match_info: Dict with details about matched columns
    """
    contrast = np.zeros(len(design_columns))
    match_info = {'positive': {}, 'negative': {}}

    # Find matching columns for positive conditions
    pos_matches = {}
    for pattern, weight in positive_weights.items():
        matching_cols = [(i, design_columns[i]) for i, col in enumerate(design_columns)
                         if pattern.lower() in col.lower()]
        if matching_cols:
            pos_matches[pattern] = ([m[0] for m in matching_cols], weight)
            match_info['positive'][pattern] = [m[1] for m in matching_cols]

    # Find matching columns for negative conditions
    neg_matches = {}
    for pattern, weight in negative_weights.items():
        matching_cols = [(i, design_columns[i]) for i, col in enumerate(design_columns)
                         if pattern.lower() in col.lower()]
        if matching_cols:
            neg_matches[pattern] = ([m[0] for m in matching_cols], weight)
            match_info['negative'][pattern] = [m[1] for m in matching_cols]

    # Calculate total weights for normalization
    total_pos_weight = sum(len(m[0]) * m[1] for m in pos_matches.values()) if pos_matches else 0
    total_neg_weight = sum(len(m[0]) * m[1] for m in neg_matches.values()) if neg_matches else 0

    # Apply positive weights (normalized to sum to 1)
    if total_pos_weight > 0:
        for pattern, (indices, weight) in pos_matches.items():
            for idx in indices:
                contrast[idx] = weight / total_pos_weight

    # Apply negative weights (normalized to sum to -1)
    if total_neg_weight > 0:
        for pattern, (indices, weight) in neg_matches.items():
            for idx in indices:
                contrast[idx] = -weight / total_neg_weight

    match_info['total_pos_weight'] = total_pos_weight
    match_info['total_neg_weight'] = total_neg_weight

    return contrast, match_info


def get_predefined_contrasts() -> dict:
    """Return dictionary of predefined contrast definitions."""
    return {
        # Goal-phase contrasts
        'goal_hit_vs_miss': 'goal_HIT > goal_MISS',
        'goal_congruent_vs_incongruent': 'goal_congruent > goal_incongruent',
        'goal_old_vs_new': 'goal_old > goal_new',
        'goal_source_hit_vs_miss': 'goal_source_hit > goal_source_miss',

        # Stim-phase contrasts
        'stim_hit_vs_miss': 'stim_HIT > stim_MISS',
        'stim_congruent_vs_incongruent': 'stim_congruent > stim_incongruent',
        'stim_old_vs_new': 'stim_old > stim_new',

        # Memory contrasts (combined)
        'remembered_vs_forgotten': 'HIT > MISS',
        'correct_vs_incorrect': 'HIT + CR > MISS + FA',
    }


# =============================================================================
# SIMPLE GLM (Standard Univariate)
# =============================================================================
def surface_1stlvl_simple(
    subID: str,
    TR: float,
    slice_time_ref: float,
    motion_threshold_scrub: float,
    motion_threshold_exclude: float,
    derivatives_root: str,
    contrasts: list[str],
    run_exclude: list[int],
    condition_column: str = 'glm_goal_conds',
    smoothed_fwhm: int = None,
    visualize: bool = False,
):
    """
    Run standard GLM with conditions pooled, then compute contrasts.

    This is the classic univariate approach where:
    - All trials of the same condition share one regressor
    - Contrasts are computed to compare conditions
    - Fixed effects combines contrasts across runs

    Args:
        subID: Subject ID
        TR: Repetition time
        slice_time_ref: Slice time reference
        motion_threshold_scrub: FD threshold for scrubbing
        motion_threshold_exclude: FD threshold for run exclusion
        derivatives_root: Path to derivatives directory
        contrasts: List of contrast definition strings
        run_exclude: List of runs to exclude
        condition_column: Column name to use for trial type conditions
        smoothed_fwhm: If specified, use pre-smoothed data with this FWHM
        visualize: Whether to plot design matrices
    """
    # Parse contrasts
    parsed_contrasts = []
    for c in contrasts:
        name, pos, neg = parse_contrast_string(c)
        parsed_contrasts.append({
            'name': name,
            'definition': c,
            'positive': pos,
            'negative': neg,
        })

    # Create output directory with timestamp
    contrast_names = '_'.join([c['name'][:20] for c in parsed_contrasts])
    output_dir = create_timestamped_output_dir(
        derivatives_root, subID, 'simple', contrast_names
    )

    print(f"\n{'='*70}")
    print(f"Simple GLM: sub-{subID}")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    if smoothed_fwhm:
        print(f"Using SMOOTHED data: {smoothed_fwhm}mm FWHM")
    else:
        print(f"Using UNSMOOTHED data")
    print(f"\nContrasts to compute:")
    for c in parsed_contrasts:
        print(f"  - {c['name']}: {c['definition']}")
    print()

    glm_params = get_glm_params(subID, TR, slice_time_ref)

    # Load surface mesh
    left_surf, right_surf = get_surface_paths(derivatives_root, subID)
    surf = PolyMesh(left=left_surf, right=right_surf)

    # Load events file
    events_path = get_events_path(derivatives_root, subID)
    events = load_events(events_path)

    # Detect all runs
    all_runs = detect_runs(derivatives_root, subID)
    print(f"Detected runs: {all_runs}")

    # Process each run
    glms = []
    events_list = []
    runs_kept = []
    runs_excluded = list(run_exclude)

    for run in all_runs:
        print(f"\n{'-'*60}")
        print(f"RUN {run}")
        print(f"{'-'*60}")

        if run in runs_excluded:
            print(f"  [EXCLUDED] A priori exclusion")
            continue

        # Load confounds
        confounds_path = get_confounds_path(derivatives_root, subID, run)
        filtered_confounds, skip_reason, motion_info = load_and_filter_confounds(
            confounds_path, motion_threshold_scrub, motion_threshold_exclude
        )

        if skip_reason:
            print(f"  [EXCLUDED] {skip_reason}")
            runs_excluded.append(run)
            continue

        # Print motion info
        print(f"  Motion Summary:")
        print(f"    Total TRs: {motion_info['total_trs']}")
        print(f"    Mean FD: {motion_info['mean_fd']:.3f} mm")
        print(f"    Max FD: {motion_info['max_fd']:.3f} mm")
        print(f"    Scrubbed TRs: {motion_info['n_scrubbed']} ({motion_info['pct_scrubbed']:.1f}%)")
        if motion_info['scrubbed_trs']:
            print(f"    Scrubbed TR indices: {motion_info['scrubbed_trs']}")

        # Load functional data
        left_data, right_data = get_functional_paths(derivatives_root, subID, run, smoothed_fwhm)
        data = PolyData(left=left_data, right=right_data)
        surface_image = SurfaceImage(mesh=surf, data=data)

        # Process events (simple: all conditions, both goal and stim)
        run_events = process_events_simple(events, run, condition_column)

        # Print detailed condition info
        cond_counts = run_events['trial_type'].value_counts().sort_index()
        print(f"\n  Events/Conditions (using column: {condition_column}):")
        print(f"    Total events: {len(run_events)}")
        for cond, count in cond_counts.items():
            print(f"    {cond}: {count} trials")

        # Fit GLM
        print(f"\n  Fitting GLM...")
        glm = FirstLevelModel(**glm_params)
        glm.fit(
            run_imgs=surface_image,
            events=run_events,
            confounds=filtered_confounds
        )

        # Print design matrix info
        dm = glm.design_matrices_[0]
        print(f"\n  Design Matrix:")
        print(f"    Shape: {dm.shape[0]} TRs x {dm.shape[1]} regressors")
        print(f"    Condition columns: {[c for c in dm.columns if not c.startswith(('drift', 'motion', 'trans', 'rot', 'frame', 'constant'))]}")
        print(f"    Nuisance columns: {len([c for c in dm.columns if c.startswith(('drift', 'motion', 'trans', 'rot', 'frame', 'constant'))])}")

        glms.append(glm)
        events_list.append(run_events)
        runs_kept.append(run)

    if not glms:
        print("No valid runs remaining. Exiting.")
        return

    # Visualize design matrix from first run
    if visualize:
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_design_matrix, show

        fig, ax = plt.subplots(figsize=(12, 10))
        plot_design_matrix(glms[0].design_matrices_[0], ax=ax)
        ax.set_title(f'Design Matrix - Run {runs_kept[0]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'design_matrix_run1.png'), dpi=150)
        show()

    # Compute contrasts for each run, then combine with fixed effects
    print(f"\n{'='*60}")
    print(f"CONTRAST COMPUTATION")
    print(f"{'='*60}")
    print(f"Computing contrasts across {len(glms)} runs...")

    contrast_results = {}

    for contrast_info in parsed_contrasts:
        first_run_shown = False  # Reset for each contrast
        contrast_name = contrast_info['name']
        print(f"\n{'-'*60}")
        print(f"Contrast: {contrast_info['definition']}")
        print(f"  Name: {contrast_name}")
        print(f"  Positive patterns: {list(contrast_info['positive'].keys())}")
        print(f"  Negative patterns: {list(contrast_info['negative'].keys())}")

        run_effect_maps = []
        run_variance_maps = []

        for run_idx, glm in enumerate(glms):
            design_cols = list(glm.design_matrices_[0].columns)

            # Build contrast vector
            contrast_vector, match_info = build_contrast_vector(
                design_cols,
                contrast_info['positive'],
                contrast_info['negative'],
                verbose=True
            )

            # Show detailed matching info for first run only
            if not first_run_shown:
                print(f"\n  Pattern Matching (Run {runs_kept[run_idx]}):")
                print(f"    POSITIVE matches (+1 total weight):")
                for pattern, matched_cols in match_info['positive'].items():
                    weight_per_col = 1.0 / len(matched_cols) if matched_cols else 0
                    print(f"      '{pattern}' -> {matched_cols}")
                    print(f"        Weight per column: {weight_per_col:.4f}")
                print(f"    NEGATIVE matches (-1 total weight):")
                for pattern, matched_cols in match_info['negative'].items():
                    weight_per_col = -1.0 / len(matched_cols) if matched_cols else 0
                    print(f"      '{pattern}' -> {matched_cols}")
                    print(f"        Weight per column: {weight_per_col:.4f}")

                # Show the full contrast vector
                print(f"\n    Contrast Vector (non-zero entries):")
                for i, (col, weight) in enumerate(zip(design_cols, contrast_vector)):
                    if weight != 0:
                        print(f"      [{i}] {col}: {weight:+.4f}")
                first_run_shown = True

            # Check if contrast is valid (has non-zero weights)
            if np.allclose(contrast_vector, 0):
                print(f"    Run {runs_kept[run_idx]}: No matching conditions, skipping")
                continue

            # Compute contrast
            contrast_result = glm.compute_contrast(
                contrast_vector,
                output_type='all'
            )

            run_effect_maps.append(contrast_result['effect_size'])
            run_variance_maps.append(contrast_result['effect_variance'])

            n_nonzero = np.sum(contrast_vector != 0)
            print(f"    Run {runs_kept[run_idx]}: Computed ({n_nonzero} regressors in contrast)")

        if not run_effect_maps:
            print(f"    WARNING: No valid runs for contrast {contrast_name}")
            continue

        # Fixed effects combination across runs
        print(f"\n  Fixed Effects Combination:")
        print(f"    Combining {len(run_effect_maps)} runs with precision-weighted averaging")
        if len(run_effect_maps) > 1:
            # Suppress expected divide-by-zero warnings (occurs at medial wall/zero-variance vertices)
            with np.errstate(divide='ignore', invalid='ignore'):
                fe_result = compute_fixed_effects(
                    contrast_imgs=run_effect_maps,
                    variance_imgs=run_variance_maps,
                    precision_weighted=True,
                )
            effect_map = fe_result[0]  # effect size
            variance_map = fe_result[1]  # variance
            # Compute z-score: effect / sqrt(variance)
            z_map = fe_result[2] if len(fe_result) > 2 else None
            print(f"    Output: effect_size, effect_variance maps")
        else:
            effect_map = run_effect_maps[0]
            variance_map = run_variance_maps[0]
            z_map = None
            print(f"    Single run - no combination needed")

        contrast_results[contrast_name] = {
            'effect_size': effect_map,
            'effect_variance': variance_map,
            'z_score': z_map,
            'definition': contrast_info['definition'],
            'n_runs': len(run_effect_maps),
        }

    # Save results
    output_data = {
        'contrast_results': contrast_results,
        'glm_type': 'simple',
        'FirstLevelModels': glms,
        'design_matrices': [g.design_matrices_[0] for g in glms],
        'total_runs': len(all_runs),
        'runs_kept': runs_kept,
        'runs_excluded': runs_excluded,
        'contrasts_defined': [c['definition'] for c in parsed_contrasts],
        'parameters': {
            'TR': TR,
            'slice_time_ref': slice_time_ref,
            'motion_threshold_scrub': motion_threshold_scrub,
            'motion_threshold_exclude': motion_threshold_exclude,
        }
    }

    output_file = os.path.join(output_dir, f'sub-{subID}_simple_glm_output.pkl')
    with open(output_file, 'wb') as f:
        pkl.dump(output_data, f)

    # Save contrast definitions as JSON for reference
    config_file = os.path.join(output_dir, 'contrast_definitions.json')
    with open(config_file, 'w') as f:
        json.dump({
            'contrasts': [{'name': c['name'], 'definition': c['definition']}
                          for c in parsed_contrasts],
            'subject': subID,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - {os.path.basename(output_file)}")
    print(f"  - contrast_definitions.json")
    print(f"{'='*70}")


# =============================================================================
# LSA/LSSM FUNCTIONS (for single-trial betas)
# =============================================================================
def create_lss_events(
    combined_events: pd.DataFrame,
    trial_number: int,
    target_event: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Create LSS events by labeling one TARGET trial uniquely."""
    events_copy = combined_events.copy()

    target_mask = events_copy['event_type'] == target_event
    target_indices = events_copy[target_mask].index.tolist()
    target_idx = target_indices[trial_number - 1]

    trial_condition = events_copy.loc[target_idx, 'trial_type']
    trial_name = f"{trial_condition}__trial_{trial_number}"
    events_copy.loc[target_idx, 'trial_type'] = trial_name

    output = events_copy[['onset', 'duration', 'trial_type']].copy()
    return output, [trial_name]


def create_lsa_events(
    combined_events: pd.DataFrame,
    target_event: str,
) -> pd.DataFrame:
    """Create LSA events by labeling each TARGET trial uniquely."""
    lsa_events = combined_events.copy()

    target_mask = lsa_events['event_type'] == target_event
    target_indices = lsa_events[target_mask].index.tolist()

    for trial_num, idx in enumerate(target_indices, start=1):
        original_type = lsa_events.loc[idx, 'trial_type']
        lsa_events.loc[idx, 'trial_type'] = f"{original_type}__{trial_num}"

    output = lsa_events[['onset', 'duration', 'trial_type']].copy()
    return output


def surface_1stlvl_LSSM(
    subID: str,
    TR: float,
    slice_time_ref: float,
    motion_threshold_scrub: float,
    motion_threshold_exclude: float,
    derivatives_root: str,
    run: int,
    event: str,
    nuisance_model: str = 'collapsed',
    condition_column: str = 'glm_goal_conds',
    smoothed_fwhm: int = None,
    output_dir: str = None,
    visualize: bool = False,
):
    """Run LSSM GLM for single-trial beta estimation."""
    print(f"\n{'='*60}")
    print(f"LSSM: sub-{subID}, run={run}, target={event}, nuisance={nuisance_model}")
    print(f"{'='*60}")

    glm_params = get_glm_params(subID, TR, slice_time_ref)

    left_surf, right_surf = get_surface_paths(derivatives_root, subID)
    surf = PolyMesh(left=left_surf, right=right_surf)

    left_data, right_data = get_functional_paths(derivatives_root, subID, run, smoothed_fwhm)
    data = PolyData(left=left_data, right=right_data)
    surface_image = SurfaceImage(mesh=surf, data=data)

    confounds_path = get_confounds_path(derivatives_root, subID, run)
    filtered_confounds, skip_reason, motion_info = load_and_filter_confounds(
        confounds_path, motion_threshold_scrub, motion_threshold_exclude
    )
    if skip_reason:
        print(f"Skipping run {run}: {skip_reason}")
        return

    if motion_info:
        print(f"Motion: Mean FD={motion_info['mean_fd']:.3f}mm, Scrubbed TRs={motion_info['n_scrubbed']}")

    events_path = get_events_path(derivatives_root, subID)
    events = load_events(events_path)
    combined_events = process_events_for_run(events, run, event, nuisance_model, condition_column)

    n_trials = get_target_event_count(combined_events, event)
    print(f"Processing {n_trials} {event} trials...")

    lss_beta_maps = []
    lss_conds = []
    lss_runs = []
    lss_design_matrices = []

    for trial_num in range(1, n_trials + 1):
        lss_events, trial_conditions = create_lss_events(combined_events, trial_num, event)

        lss_glm = FirstLevelModel(**glm_params)
        lss_glm.fit(
            run_imgs=surface_image,
            events=lss_events,
            confounds=filtered_confounds
        )

        lss_design_matrices.append(lss_glm.design_matrices_[0])

        for cond in trial_conditions:
            beta_map = lss_glm.compute_contrast(cond, output_type="all")
            condition_name = cond.split("__")[0]
            lss_beta_maps.append(beta_map)
            lss_conds.append(condition_name)
            lss_runs.append(run)

    # Use provided output_dir or create default
    if output_dir is None:
        output_dir = get_output_dir(derivatives_root, subID)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'beta_maps': lss_beta_maps,
        'conditions': lss_conds,
        'runs': lss_runs,
        'design_matrices': lss_design_matrices,
        'nuisance_model': nuisance_model,
    }

    output_file = os.path.join(
        output_dir,
        f'sub-{subID}_run-{run:02d}_event-{event}_nuisance-{nuisance_model}_LSSM_output.pkl'
    )

    with open(output_file, 'wb') as f:
        pkl.dump(output_data, f)

    print(f"LSSM output saved to {output_file}")


def surface_1stlvl_LSA(
    subID: str,
    TR: float,
    slice_time_ref: float,
    motion_threshold_scrub: float,
    motion_threshold_exclude: float,
    derivatives_root: str,
    event: str,
    nuisance_model: str,
    run_exclude: list[int],
    condition_column: str = 'glm_goal_conds',
    smoothed_fwhm: int = None,
    output_dir: str = None,
    visualize: bool = False,
):
    """Run LSA GLM for single-trial beta estimation."""
    print(f"\n{'='*60}")
    print(f"LSA: sub-{subID}, target={event}, nuisance={nuisance_model}")
    print(f"{'='*60}")

    glm_params = get_glm_params(subID, TR, slice_time_ref)

    left_surf, right_surf = get_surface_paths(derivatives_root, subID)
    surf = PolyMesh(left=left_surf, right=right_surf)

    events_path = get_events_path(derivatives_root, subID)
    events = load_events(events_path)

    all_runs = detect_runs(derivatives_root, subID)
    print(f"Detected runs: {all_runs}")

    glms = []
    events_list = []
    combined_events_list = []
    runs_kept = []
    runs_excluded = list(run_exclude)

    for run in all_runs:
        if run in runs_excluded:
            print(f"Run {run} excluded (a priori)")
            continue

        confounds_path = get_confounds_path(derivatives_root, subID, run)
        filtered_confounds, skip_reason, motion_info = load_and_filter_confounds(
            confounds_path, motion_threshold_scrub, motion_threshold_exclude
        )

        if skip_reason:
            print(f"Run {run} excluded: {skip_reason}")
            runs_excluded.append(run)
            continue

        left_data, right_data = get_functional_paths(derivatives_root, subID, run, smoothed_fwhm)
        data = PolyData(left=left_data, right=right_data)
        surface_image = SurfaceImage(mesh=surf, data=data)

        combined_events = process_events_for_run(events, run, event, nuisance_model, condition_column)
        lsa_events = create_lsa_events(combined_events, event)

        n_target = get_target_event_count(combined_events, event)
        print(f"Run {run} - {n_target} {event} trials (scrubbed TRs: {motion_info['n_scrubbed']})")

        glm = FirstLevelModel(**glm_params)
        glm.fit(
            run_imgs=surface_image,
            events=lsa_events,
            confounds=filtered_confounds
        )

        glms.append(glm)
        events_list.append(lsa_events)
        combined_events_list.append(combined_events)
        runs_kept.append(run)

    if not glms:
        print("No valid runs remaining. Exiting.")
        return

    lsa_beta_maps = []
    lsa_conds = []
    run_list = []

    for run_idx, glm in enumerate(glms):
        run_events = events_list[run_idx]
        target_conds = [c for c in run_events['trial_type'].unique() if '__' in c]

        for cond in target_conds:
            contrast = np.zeros(glm.design_matrices_[0].shape[1])
            col_idx = glm.design_matrices_[0].columns.get_loc(cond)
            contrast[col_idx] = 1

            beta_map = glm.compute_contrast(contrast, output_type="all")
            run_list.append(runs_kept[run_idx])
            lsa_beta_maps.append(beta_map)
            lsa_conds.append(cond)

    if output_dir is None:
        output_dir = get_output_dir(derivatives_root, subID)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'beta_maps': lsa_beta_maps,
        'conditions': lsa_conds,
        'FirstLevelModel': glms,
        'total_runs': len(all_runs),
        'run_excluded': runs_excluded,
        'runs': run_list,
        'nuisance_model': nuisance_model,
    }

    output_file = os.path.join(
        output_dir,
        f'sub-{subID}_event-{event}_nuisance-{nuisance_model}_LSA_output.pkl'
    )

    with open(output_file, 'wb') as f:
        pkl.dump(output_data, f)

    print(f"LSA output saved to {output_file}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_run_exclude(s: str) -> list[int]:
    """Parse run exclusion string."""
    if not s:
        return []
    tokens = str(s).replace(',', ' ').split()
    return [int(t) for t in tokens if t.isdigit()]


def main():
    parser = argparse.ArgumentParser(
        description='Surface-based first-level GLM analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GLM Types:
  simple  - Standard univariate GLM with conditions pooled. Use for contrasts.
  LSA     - Least Squares All: single-trial betas, all in one model.
  LSSM    - Least Squares Separate: single-trial betas, separate models.

Contrast Definition (for simple GLM):
  Use --contrast to specify contrasts. Multiple --contrast flags allowed.

  Format: "CONDITION_A > CONDITION_B" or "CONDITION_A - CONDITION_B"

  Examples:
    --contrast "goal_HIT > goal_MISS"
    --contrast "stim_congruent > stim_incongruent"
    --contrast "HIT + CR > MISS + FA"

  Condition matching is partial and case-insensitive:
    "HIT" matches "goal_congruent_old_HIT", "stim_incongruent_old_HIT", etc.

  Predefined contrasts (use --list-contrasts to see all):
    --contrast goal_hit_vs_miss
    --contrast remembered_vs_forgotten

Examples:
  # Simple GLM with contrast
  python surface_1stlvl_analysis.py --subID 073 --GLM_type simple \\
      --contrast "goal_HIT > goal_MISS"

  # Multiple contrasts
  python surface_1stlvl_analysis.py --subID 073 --GLM_type simple \\
      --contrast "goal_HIT > goal_MISS" \\
      --contrast "stim_congruent > stim_incongruent"

  # LSA for single-trial betas
  python surface_1stlvl_analysis.py --subID 073 --GLM_type LSA --event stim

  # List predefined contrasts
  python surface_1stlvl_analysis.py --list-contrasts
        """
    )

    parser.add_argument('--subID', type=str,
                        help='Subject ID without "sub-" prefix')
    parser.add_argument('--GLM_type', type=str, choices=['simple', 'LSA', 'LSSM'],
                        help='GLM type')

    # Simple GLM options
    parser.add_argument('--contrast', type=str, action='append', dest='contrasts',
                        help='Contrast definition (can specify multiple)')
    parser.add_argument('--contrast-file', type=str,
                        help='JSON file with contrast definitions')
    parser.add_argument('--list-contrasts', action='store_true',
                        help='List predefined contrasts and exit')

    # LSA/LSSM options
    parser.add_argument('--event', type=str, choices=['goal', 'stim'],
                        help='Target event type (for LSA/LSSM)')
    parser.add_argument('--nuisance_model', type=str, default='collapsed',
                        choices=['collapsed', 'by_condition'],
                        help='Nuisance model for LSA/LSSM')

    # Common options
    parser.add_argument('--TR', type=float, default=2.0)
    parser.add_argument('--slice_time_ref', type=float, default=0.5)
    parser.add_argument('--motion_threshold_scrub', type=float, default=0.9)
    parser.add_argument('--motion_threshold_exclude', type=float, default=5.0)
    parser.add_argument('--derivatives_root', type=str,
                        default='/Users/shawnschwartz/Developer/datasci-homelab/volumes/home/work/amass/derivatives')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--run_excluded', type=str, default='')
    parser.add_argument('--condition_column', type=str, default='glm_goal_conds',
                        help='Column name for trial type conditions (default: glm_goal_conds)')
    parser.add_argument('--smoothed_fwhm', type=int, default=None,
                        help='Use pre-smoothed data with this FWHM in mm (e.g., 4 for 4mm smoothed)')

    args = parser.parse_args()

    # Handle --list-contrasts
    if args.list_contrasts:
        print("\nPredefined Contrasts:")
        print("-" * 60)
        for name, definition in get_predefined_contrasts().items():
            print(f"  {name}:")
            print(f"    {definition}")
        print()
        return

    # Validate arguments
    if not args.subID:
        parser.error("--subID is required")
    if not args.GLM_type:
        parser.error("--GLM_type is required")

    run_exclude = parse_run_exclude(args.run_excluded)

    print(f"\n{'#'*70}")
    print(f"Surface 1st-Level Analysis")
    print(f"{'#'*70}")
    print(f"Subject: sub-{args.subID}")
    print(f"GLM Type: {args.GLM_type}")
    print(f"TR: {args.TR}s")
    if args.smoothed_fwhm:
        print(f"Data: Pre-smoothed ({args.smoothed_fwhm}mm FWHM)")
    else:
        print(f"Data: Unsmoothed (fMRIPrep output)")
    if run_exclude:
        print(f"Runs excluded: {run_exclude}")
    print(f"{'#'*70}\n")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        if args.GLM_type == 'simple':
            # Gather contrasts
            contrasts = args.contrasts or []

            # Add contrasts from file
            if args.contrast_file:
                file_contrasts = load_contrast_file(args.contrast_file)
                contrasts.extend([c['definition'] for c in file_contrasts])

            # Expand predefined contrast names
            predefined = get_predefined_contrasts()
            expanded = []
            for c in contrasts:
                if c in predefined:
                    expanded.append(predefined[c])
                else:
                    expanded.append(c)
            contrasts = expanded

            if not contrasts:
                parser.error("--contrast or --contrast-file required for simple GLM")

            surface_1stlvl_simple(
                subID=args.subID,
                TR=args.TR,
                slice_time_ref=args.slice_time_ref,
                motion_threshold_scrub=args.motion_threshold_scrub,
                motion_threshold_exclude=args.motion_threshold_exclude,
                derivatives_root=args.derivatives_root,
                contrasts=contrasts,
                run_exclude=run_exclude,
                condition_column=args.condition_column,
                smoothed_fwhm=args.smoothed_fwhm,
                visualize=args.visualize,
            )

        elif args.GLM_type == 'LSA':
            if not args.event:
                parser.error("--event required for LSA")

            output_dir = create_timestamped_output_dir(
                args.derivatives_root, args.subID, 'LSA', f'{args.event}_{args.nuisance_model}'
            )

            surface_1stlvl_LSA(
                subID=args.subID,
                TR=args.TR,
                slice_time_ref=args.slice_time_ref,
                motion_threshold_scrub=args.motion_threshold_scrub,
                motion_threshold_exclude=args.motion_threshold_exclude,
                derivatives_root=args.derivatives_root,
                event=args.event,
                nuisance_model=args.nuisance_model,
                run_exclude=run_exclude,
                condition_column=args.condition_column,
                smoothed_fwhm=args.smoothed_fwhm,
                output_dir=output_dir,
                visualize=args.visualize,
            )

        elif args.GLM_type == 'LSSM':
            if not args.event:
                parser.error("--event required for LSSM")

            output_dir = create_timestamped_output_dir(
                args.derivatives_root, args.subID, 'LSSM', f'{args.event}_{args.nuisance_model}'
            )

            runs = detect_runs(args.derivatives_root, args.subID)
            runs = [r for r in runs if r not in run_exclude]

            for run in runs:
                surface_1stlvl_LSSM(
                    subID=args.subID,
                    TR=args.TR,
                    slice_time_ref=args.slice_time_ref,
                    motion_threshold_scrub=args.motion_threshold_scrub,
                    motion_threshold_exclude=args.motion_threshold_exclude,
                    derivatives_root=args.derivatives_root,
                    run=run,
                    event=args.event,
                    nuisance_model=args.nuisance_model,
                    condition_column=args.condition_column,
                    smoothed_fwhm=args.smoothed_fwhm,
                    output_dir=output_dir,
                    visualize=args.visualize,
                )

        for w in caught_warnings:
            print(f"Warning: {w.message}")

    print("\nDone!")


if __name__ == "__main__":
    main()
