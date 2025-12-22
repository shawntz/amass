#!/bin/bash

# AMASS retrieval batch eyeris epoch preprocessing slurm submission script

set -e  # exit on error

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [subject]"
    echo "  subject: Subject ID (e.g., sub-1021; optional - for running single ids)"
    exit 1
fi

SUBJECT=${1:-""}  # optional

if [ -z "$SUBJECT" ]; then  # case when no subject provided by user
    SUBJECT_LIST=$(mktemp -p /oak/stanford/groups/awagner/yaams-haams/eyeris-workbench)
    make subjects-list > "$SUBJECT_LIST"
    NUM_SUBJECTS=$(wc -l < "$SUBJECT_LIST")
    echo "No subject specified, running all $NUM_SUBJECTS subjects from sourcedata/eye"
else
    SUBJECT_LIST=$(mktemp -p /oak/stanford/groups/awagner/yaams-haams/eyeris-workbench)
    echo "$SUBJECT" > "$SUBJECT_LIST"
    echo "Running single subject: $SUBJECT"
fi

JOB_NAME="eyeris"

command -v pip3 >/dev/null || module load python/3.9.0
python3 -c "import yaml" 2>/dev/null || python3 -m pip install --user pyyaml

get_yaml_value() {
    local key=$1
    python3 -c "
import yaml
import sys
with open('/oak/stanford/groups/awagner/yaams-haams/eyeris-workbench/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
value = config
for k in sys.argv[1].split('.'):
    value = value[k]
print(value)
" "$key"
}

PROJECT_BASE_DIR=$(get_yaml_value "project.base_dir")
PROJECT_USER_EMAIL=$(get_yaml_value "project.user_email")
SLURM_EMAIL=$(get_yaml_value "slurm.email")
SLURM_TIME=$(get_yaml_value "slurm.time")
SLURM_MEM=$(get_yaml_value "slurm.mem_per_cpu")
SLURM_CPUS=$(get_yaml_value "slurm.cpus_per_task")
SLURM_PARTITION=$(get_yaml_value "slurm.partition")
SLURM_ARRAY_THROTTLE=$(get_yaml_value "slurm.array_throttle")
SLURM_LOG_DIR=$(get_yaml_value "slurm.log_dir")

PROJECT_BASE_DIR=$(echo "$PROJECT_BASE_DIR" | sed "s|\${project.base_dir}|$PROJECT_BASE_DIR|g")
PROJECT_USER_EMAIL=$(echo "$PROJECT_USER_EMAIL" | sed "s|\${project.user_email}|$PROJECT_USER_EMAIL|g")
SLURM_EMAIL=$(echo "$SLURM_EMAIL" | sed "s|\${project.user_email}|$PROJECT_USER_EMAIL|g")
SLURM_LOG_DIR=$(echo "$SLURM_LOG_DIR" | sed "s|\${project.base_dir}|$PROJECT_BASE_DIR|g")

mkdir -p "${SLURM_LOG_DIR}/${JOB_NAME}"

if [ "$SESSION" = "all" ]; then
    ARRAY_SIZE=$((NUM_SUBJECTS * 2))
else
    ARRAY_SIZE=$NUM_SUBJECTS
fi

array_opt="--array=0-$((ARRAY_SIZE-1))%${SLURM_ARRAY_THROTTLE}"

echo "($(date)) [INFO] - BATCH SLURM JOB PARAMETERS:"
echo "  --job-name: ${JOB_NAME}"
echo "  ${array_opt}"
echo "  --time: ${SLURM_TIME}"
echo "  --mem-per-cpu: ${SLURM_MEM}"
echo "  --cpus-per-task: ${SLURM_CPUS}"
echo "  --partition: ${SLURM_PARTITION}"
echo "  --output: ${SLURM_LOG_DIR}/${JOB_NAME}/slurm/%x_%A_%a.out"
echo "  --error: ${SLURM_LOG_DIR}/${JOB_NAME}/slurm/%x_%A_%a.err"
echo "  --mail-type: BEGIN,END,FAIL"
echo "  --mail-user: ${SLURM_EMAIL}"
echo "  Subject range: ${START_SUBJECT} to ${END_SUBJECT} (${NUM_SUBJECTS} subjects)"
echo "  Array size: ${ARRAY_SIZE}"

# submit job
sbatch \
    --job-name=${JOB_NAME} \
    ${array_opt} \
    --time=${SLURM_TIME} \
    --mem-per-cpu=${SLURM_MEM} \
    --cpus-per-task=${SLURM_CPUS} \
    --partition=${SLURM_PARTITION} \
    --output=${SLURM_LOG_DIR}/${JOB_NAME}/slurm/%x_%A_%a.out \
    --error=${SLURM_LOG_DIR}/${JOB_NAME}/slurm/%x_%A_%a.err \
    --mail-type=BEGIN,END,FAIL \
    --mail-user=${SLURM_EMAIL} \
    ${PROJECT_BASE_DIR}/cmd/run-eyeris ${JOB_NAME} ${SUBJECT_LIST}

echo "($(date)) [INFO] - Batch job submitted successfully"
echo "($(date)) [INFO] - Subject list file: ${SUBJECT_LIST}"
echo "($(date)) [INFO] - Note: Temp file will be cleaned up manually or by system"
