#!/bin/bash
set -e  # Exit on error

# Load common logging functions
source utils/logging.sh

# Initialize logging
LOG_DIR="logs/corruption"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/corruption_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error in model corruption step at line $line_number"
    log_error "Exit code: $exit_code"
    
    # Log node-specific information
    log_error "Node: $(hostname)"
    log_error "Memory usage: $(free -h)"
    log_error "CPU usage: $(top -bn1 | head -n 5)"
    
    # Log SLURM-specific information
    if [ ! -z "$SLURM_JOB_ID" ]; then
        log_error "SLURM job details:"
        log_error "$(scontrol show job $SLURM_JOB_ID)"
        log_error "Node list: $SLURM_JOB_NODELIST"
        log_error "Task distribution: $(scontrol show hostname $SLURM_JOB_NODELIST)"
    fi
    
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# Validate environment
if ! command -v python &> /dev/null; then
    log_error "Python not found"
    exit 1
fi

if ! command -v pytorch &> /dev/null; then
    log_warning "PyTorch module not loaded, attempting to load..."
    module load pytorch/2.0.1 || {
        log_error "Failed to load PyTorch module"
        exit 1
    }
fi

# Validate input directories
MODELS=(
    "models/lmp_sw_mos2.pth"
    "models/lmp_sw_wse2.pth"
    "models/lmp_kc_wse2_mos2.pth"
)

for MODEL in "${MODELS[@]}"; do
    if [ ! -f "$MODEL" ]; then
        log_error "Model file not found: $MODEL"
        exit 1
    fi
done

# Create output directories
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR="corrupted_models/${MODEL_NAME%.*}"
    mkdir -p "$OUTPUT_DIR" || {
        log_error "Failed to create output directory: $OUTPUT_DIR"
        exit 1
    }
done

# Function to monitor job progress
monitor_job() {
    local job_id=$1
    local model=$2
    
    while true; do
        local state=$(scontrol show job $job_id | grep -oP "JobState=\K\w+")
        case $state in
            COMPLETED)
                log_info "Job $job_id completed successfully for model $model"
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                log_error "Job $job_id failed for model $model with state: $state"
                return 1
                ;;
            RUNNING)
                log_info "Job $job_id still running for model $model..."
                sleep 60
                ;;
            *)
                log_warning "Unknown job state: $state for job $job_id"
                sleep 60
                ;;
        esac
    done
}

# Launch jobs
active_jobs=()
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR="corrupted_models/${MODEL_NAME%.*}"
    log_info "Processing model: $MODEL"
    
    for SEED in {0..9}; do
        log_info "Launching job for seed $SEED"
        job_id=$(sbatch --parsable --exclusive -n 1 \
            python 1_generate_corrupted_models.py \
            --model "$MODEL" \
            --output-dir "$OUTPUT_DIR" \
            --seed "$SEED")
        
        if [ $? -ne 0 ]; then
            log_error "Failed to launch job for model $MODEL seed $SEED"
            exit 1
        fi
        
        active_jobs+=("$job_id:$MODEL")
        log_info "Launched job $job_id for model $MODEL seed $SEED"
    done
done

# Monitor jobs
failed_jobs=0
for job_info in "${active_jobs[@]}"; do
    job_id=${job_info%:*}
    model=${job_info#*:}
    
    if ! monitor_job $job_id $model; then
        ((failed_jobs++))
        log_error "Job $job_id failed for model $model"
    fi
done

# Check results
log_info "Checking results..."
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR="corrupted_models/${MODEL_NAME%.*}"
    
    # Count expected files
    expected_files=$((10 * 12))  # 10 seeds * 12 corruption factors
    actual_files=$(find "$OUTPUT_DIR" -name "corruptfac_*.pth" | wc -l)
    
    if [ "$actual_files" -ne "$expected_files" ]; then
        log_error "Missing files for model $MODEL: expected $expected_files, found $actual_files"
        ((failed_jobs++))
    fi
done

# Final status
if [ $failed_jobs -gt 0 ]; then
    log_error "Workflow completed with $failed_jobs failed jobs"
    exit 1
else
    log_info "All jobs completed successfully"
fi

# Archive logs
log_info "Archiving logs..."
tar -czf "${LOG_DIR}/corruption_${TIMESTAMP}_logs.tar.gz" "$LOG_FILE"
