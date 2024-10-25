#!/bin/bash
set -e  # Exit on error

# Load common logging functions
source utils/logging.sh

# Initialize logging
LOG_DIR="logs/relaxation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/relaxation_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error in structure relaxation step at line $line_number"
    log_error "Exit code: $exit_code"
    
    # Log node-specific information
    log_error "Node: $(hostname)"
    log_error "Memory usage: $(free -h)"
    log_error "CPU usage: $(top -bn1 | head -n 5)"
    log_error "GPU status: $(nvidia-smi 2>/dev/null || echo 'No GPU available')"
    
    # Log SLURM-specific information
    if [ ! -z "$SLURM_JOB_ID" ]; then
        log_error "SLURM job details:"
        log_error "$(scontrol show job $SLURM_JOB_ID)"
        log_error "Node list: $SLURM_JOB_NODELIST"
        log_error "CPU allocation: $SLURM_CPUS_PER_TASK CPUs per task"
        log_error "Memory allocation: $SLURM_MEM_PER_NODE MB per node"
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

# Validate input directories and files
STRUCTURES=(
    "structures/MoS2_WSe2_1D.xyz"
    "structures/MoS2_WSe2_2D.xyz"
)

for STRUCT in "${STRUCTURES[@]}"; do
    if [ ! -f "$STRUCT" ]; then
        log_error "Structure file not found: $STRUCT"
        exit 1
    fi
done

# Check for corrupted models
if [ ! -d "corrupted_models" ]; then
    log_error "Corrupted models directory not found"
    exit 1
fi

# Function to monitor job progress
monitor_job() {
    local job_id=$1
    local structure=$2
    local seed=$3
    local idx=$4
    
    while true; do
        local state=$(scontrol show job $job_id | grep -oP "JobState=\K\w+")
        case $state in
            COMPLETED)
                log_info "Job $job_id completed successfully for structure $structure (seed: $seed, idx: $idx)"
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                log_error "Job $job_id failed for structure $structure with state: $state"
                log_error "Seed: $seed, Index: $idx"
                return 1
                ;;
            RUNNING)
                log_info "Job $job_id still running for structure $structure..."
                sleep 60
                ;;
            *)
                log_warning "Unknown job state: $state for job $job_id"
                sleep 60
                ;;
        esac
    done
}

# Function to verify output files
verify_outputs() {
    local structure=$1
    local seed=$2
    local idx=$3
    local output_dir=$4
    
    local base_name="relaxed_SEED_${seed}_idx_${idx}"
    local expected_files=(
        "${output_dir}/${base_name}.traj"
        "${output_dir}/${base_name}.traj.xyz"
        "${output_dir}/${base_name}_lowest_energy.xyz"
    )
    
    for file in "${expected_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Missing output file: $file"
            return 1
        fi
        
        # Check file size
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
        if [ "$size" -eq 0 ]; then
            log_error "Empty output file: $file"
            return 1
        fi
    done
    
    return 0
}

# Launch jobs
active_jobs=()
for STRUCT in "${STRUCTURES[@]}"; do
    STRUCT_NAME=$(basename "$STRUCT" .xyz)
    OUTPUT_DIR="relaxed_structures/${STRUCT_NAME}"
    mkdir -p "$OUTPUT_DIR" || {
        log_error "Failed to create output directory: $OUTPUT_DIR"
        exit 1
    }
    
    log_info "Processing structure: $STRUCT"
    
    for SEED in {0..9}; do
        for IDX in {0..11}; do
            # Check if output already exists and is valid
            if verify_outputs "$STRUCT" "$SEED" "$IDX" "$OUTPUT_DIR"; then
                log_info "Outputs already exist for $STRUCT (seed: $SEED, idx: $IDX), skipping..."
                continue
            fi
            
            log_info "Launching job for seed $SEED, index $IDX"
            job_id=$(sbatch --parsable --exclusive -n 1 --cpus-per-task=8 \
                python 2_relax_structures.py \
                --structure "$STRUCT" \
                --output-dir "$OUTPUT_DIR" \
                --seed "$SEED" \
                --corruption-idx "$IDX" \
                --device "cpu")
            
            if [ $? -ne 0 ]; then
                log_error "Failed to launch job for structure $STRUCT (seed: $SEED, idx: $IDX)"
                exit 1
            fi
            
            active_jobs+=("$job_id:$STRUCT:$SEED:$IDX")
            log_info "Launched job $job_id"
        done
    done
done

# Monitor jobs
failed_jobs=0
for job_info in "${active_jobs[@]}"; do
    IFS=':' read -r job_id structure seed idx <<< "$job_info"
    
    if ! monitor_job "$job_id" "$structure" "$seed" "$idx"; then
        ((failed_jobs++))
        log_error "Job $job_id failed"
        continue
    fi
    
    # Verify outputs after successful completion
    STRUCT_NAME=$(basename "$structure" .xyz)
    OUTPUT_DIR="relaxed_structures/${STRUCT_NAME}"
    if ! verify_outputs "$structure" "$seed" "$idx" "$OUTPUT_DIR"; then
        ((failed_jobs++))
        log_error "Output verification failed for job $job_id"
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
tar -czf "${LOG_DIR}/relaxation_${TIMESTAMP}_logs.tar.gz" "$LOG_FILE"
