#!/bin/bash

# Master script to run the entire workflow
set -e  # Exit on error

# Load logging functions
source scripts/utils/logging.sh

# Initialize logging
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/workflow_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Logging functions
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING] $*" | tee -a "$LOG_FILE"
}

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error occurred in script at line $line_number"
    log_error "Exit code: $exit_code"
    
    # Collect system information
    log_error "System information:"
    log_error "$(uname -a)"
    log_error "Available memory: $(free -h)"
    log_error "Disk space: $(df -h)"
    
    # Check SLURM job status if applicable
    if [ ! -z "$SLURM_JOB_ID" ]; then
        log_error "SLURM job information:"
        log_error "$(scontrol show job $SLURM_JOB_ID)"
    fi
    
    exit $exit_code
}

# Set up error handling
trap 'handle_error ${LINENO}' ERR

# Default settings
START_STEP=1
END_STEP=4
ACCOUNT="m3606"
PARTITION="regular"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-step)
            START_STEP=$2
            shift 2
            ;;
        --end-step)
            END_STEP=$2
            shift 2
            ;;
        --account)
            ACCOUNT=$2
            shift 2
            ;;
        --partition)
            PARTITION=$2
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input parameters
if [ $START_STEP -gt $END_STEP ]; then
    log_error "Start step ($START_STEP) cannot be greater than end step ($END_STEP)"
    exit 1
fi

# Create necessary directories
for dir in logs models corrupted_models relaxed_structures results processed_data plots; do
    if ! mkdir -p "$dir"; then
        log_error "Failed to create directory: $dir"
        exit 1
    fi
done

# Enhanced job status checking
check_job_status() {
    local job_id=$1
    local step_name=$2
    local max_attempts=30
    local wait_time=60
    
    log_info "Waiting for job $job_id ($step_name) to complete..."
    
    for ((i=1; i<=max_attempts; i++)); do
        local job_state=$(scontrol show job $job_id | grep -oP "JobState=\K\w+")
        
        case $job_state in
            COMPLETED)
                log_info "Job $job_id ($step_name) completed successfully"
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                log_error "Job $job_id ($step_name) failed with state: $job_state"
                log_error "Job details:"
                scontrol show job $job_id | tee -a "$LOG_FILE"
                return 1
                ;;
            PENDING|RUNNING)
                if [ $i -eq $max_attempts ]; then
                    log_error "Timeout waiting for job $job_id ($step_name)"
                    return 1
                fi
                sleep $wait_time
                ;;
            *)
                log_warning "Unknown job state: $job_state"
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    local step=$1
    
    case $step in
        2)
            if [ ! -d "corrupted_models" ] || [ -z "$(ls -A corrupted_models)" ]; then
                log_error "Corrupted models not found. Run step 1 first."
                return 1
            fi
            ;;
        3)
            if [ ! -d "relaxed_structures" ] || [ -z "$(ls -A relaxed_structures)" ]; then
                log_error "Relaxed structures not found. Run step 2 first."
                return 1
            fi
            ;;
        4)
            if [ ! -d "results" ] || [ -z "$(ls -A results)" ]; then
                log_error "Distance results not found. Run step 3 first."
                return 1
            fi
            ;;
    esac
    return 0
}

# Run workflow steps
for step in $(seq $START_STEP $END_STEP); do
    log_info "Starting step $step"
    
    # Check prerequisites
    check_prerequisites $step
    
    case $step in
        1)
            log_info "Generating corrupted models..."
            job_id=$(sbatch --parsable \
                --account=$ACCOUNT \
                --partition=$PARTITION \
                scripts/1_run_corruption.sh)
            ;;
        2)
            log_info "Relaxing structures..."
            job_id=$(sbatch --parsable \
                --account=$ACCOUNT \
                --partition=$PARTITION \
                --dependency=afterok:$prev_job_id \
                scripts/2_run_relaxation.sh)
            ;;
        3)
            log_info "Calculating distances..."
            job_id=$(sbatch --parsable \
                --account=$ACCOUNT \
                --partition=$PARTITION \
                --dependency=afterok:$prev_job_id \
                scripts/3_run_distance_calculation.sh)
            ;;
        4)
            log_info "Processing results and creating plots..."
            job_id=$(sbatch --parsable \
                --account=$ACCOUNT \
                --partition=$PARTITION \
                --dependency=afterok:$prev_job_id \
                scripts/4_run_analysis.sh)
            ;;
    esac
    
    check_job_status $job_id "step $step"
    prev_job_id=$job_id
done

# Print summary
log_info "Workflow completed successfully!"
log_info "Summary of results:"
log_info "==================="
log_info "Corrupted models: $(ls -1 corrupted_models | wc -l)"
log_info "Relaxed structures: $(ls -1 relaxed_structures | wc -l)"
log_info "Distance results: $(ls -1 results | wc -l)"
log_info "Generated plots: $(ls -1 plots | wc -l)"

# Archive logs
log_info "Archiving logs..."
tar -czf "logs/workflow_${TIMESTAMP}_logs.tar.gz" "$LOG_FILE"
