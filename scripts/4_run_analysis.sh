#!/bin/bash
set -e  # Exit on error

# Load common logging functions
source utils/logging.sh

# Initialize logging
LOG_DIR="logs/analysis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/analysis_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error in analysis step at line $line_number"
    log_error "Exit code: $exit_code"
    
    # Log node-specific information
    log_error "Node: $(hostname)"
    log_error "Memory usage: $(free -h)"
    log_error "CPU usage: $(top -bn1 | head -n 5)"
    log_error "Python processes: $(ps aux | grep python)"
    
    # Log SLURM-specific information
    if [ ! -z "$SLURM_JOB_ID" ]; then
        log_error "SLURM job details:"
        log_error "$(scontrol show job $SLURM_JOB_ID)"
        log_error "Node list: $SLURM_JOB_NODELIST"
        log_error "CPU allocation: $SLURM_CPUS_PER_TASK CPUs per task"
    fi
    
    exit $exit_code
}

trap 'handle_error ${LINENO}' ERR

# Validate environment
if ! command -v python &> /dev/null; then
    log_error "Python not found"
    exit 1
fi

# Check for required Python packages
python -c "import numpy; import matplotlib" || {
    log_error "Required Python packages not found"
    exit 1
}

# Create directories
mkdir -p processed_data plots || {
    log_error "Failed to create output directories"
    exit 1
}

# Function to verify input files
verify_input_files() {
    local results_dir="results"
    local required_files=(
        "MoS2_WSe2_1D/interlayer/distances.npy"
        "MoS2_WSe2_1D/intralayer/distances.npy"
        "MoS2_WSe2_2D/interlayer/distances.npy"
        "MoS2_WSe2_2D/intralayer/distances.npy"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "${results_dir}/${file}" ]; then
            log_error "Required input file not found: ${results_dir}/${file}"
            return 1
        fi
        
        # Check file size
        local size=$(stat -f%z "${results_dir}/${file}" 2>/dev/null || stat -c%s "${results_dir}/${file}")
        if [ "$size" -eq 0 ]; then
            log_error "Empty input file: ${results_dir}/${file}"
            return 1
        fi
    done
    
    return 0
}

# Function to monitor job progress
monitor_job() {
    local job_id=$1
    local step_name=$2
    
    while true; do
        local state=$(scontrol show job $job_id | grep -oP "JobState=\K\w+")
        case $state in
            COMPLETED)
                log_info "Job $job_id completed successfully for $step_name"
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                log_error "Job $job_id failed for $step_name with state: $state"
                return 1
                ;;
            RUNNING)
                log_info "Job $job_id still running for $step_name..."
                sleep 30
                ;;
            *)
                log_warning "Unknown job state: $state for job $job_id"
                sleep 30
                ;;
        esac
    done
}

# Verify input files
if ! verify_input_files; then
    exit 1
fi

# Process distances
log_info "Processing distances..."
job_id=$(sbatch --parsable --cpus-per-task=4 \
    python 4_process_distances.py \
    --input-dir results \
    --output-dir processed_data)

if ! monitor_job "$job_id" "distance processing"; then
    log_error "Distance processing failed"
    exit 1
fi

# Create plots
log_info "Creating plots..."
job_id=$(sbatch --parsable --cpus-per-task=4 \
    --dependency=afterok:$job_id \
    python 5_create_plots.py \
    --input-dir processed_data \
    --output-dir plots \
    --style matplotlib.rc)

if ! monitor_job "$job_id" "plot creation"; then
    log_error "Plot creation failed"
    exit 1
fi

# Verify outputs
log_info "Verifying outputs..."
expected_plots=(
    "interlayer_distances_1D.pdf"
    "intralayer_distances_1D.pdf"
    "interlayer_distances_2D.pdf"
    "intralayer_distances_2D.pdf"
    "interlayer_1D_vs_2D_comparison.pdf"
    "intralayer_1D_vs_2D_comparison.pdf"
)

failed_verifications=0
for plot in "${expected_plots[@]}"; do
    if [ ! -f "plots/${plot}" ]; then
        log_error "Expected plot not found: plots/${plot}"
        ((failed_verifications++))
    fi
done

if [ $failed_verifications -gt 0 ]; then
    log_error "Output verification failed"
    exit 1
fi

log_info "Analysis completed successfully"

# Archive logs
log_info "Archiving logs..."
tar -czf "${LOG_DIR}/analysis_${TIMESTAMP}_logs.tar.gz" "$LOG_FILE"
