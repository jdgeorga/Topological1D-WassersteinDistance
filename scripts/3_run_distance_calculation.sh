#!/bin/bash
set -e  # Exit on error

# Load common logging functions
source utils/logging.sh

# Initialize logging
LOG_DIR="logs/distances"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/distances_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Error handler
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Error in distance calculation step at line $line_number"
    log_error "Exit code: $exit_code"
    
    # Log node-specific information
    log_error "Node: $(hostname)"
    log_error "Memory usage: $(free -h)"
    log_error "CPU usage: $(top -bn1 | head -n 5)"
    log_error "Available CPUs: $(nproc)"
    
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

# Function to verify input directory structure
verify_input_structure() {
    local struct_type=$1
    local analysis_type=$2
    local relaxed_dir="relaxed_structures/${struct_type}"
    
    if [ ! -d "$relaxed_dir" ]; then
        log_error "Relaxed structures directory not found: $relaxed_dir"
        return 1
    fi
    
    # Check for expected files
    local file_count=$(find "$relaxed_dir" -name "*_lowest_energy.xyz" | wc -l)
    local expected_count=$((10 * 12))  # 10 seeds * 12 corruption factors
    
    if [ "$file_count" -lt "$expected_count" ]; then
        log_warning "Found fewer files than expected in $relaxed_dir: $file_count/$expected_count"
    fi
    
    return 0
}

# Function to monitor job progress
monitor_job() {
    local job_id=$1
    local struct_type=$2
    local analysis_type=$3
    
    while true; do
        local state=$(scontrol show job $job_id | grep -oP "JobState=\K\w+")
        case $state in
            COMPLETED)
                log_info "Job $job_id completed successfully for ${struct_type}/${analysis_type}"
                return 0
                ;;
            FAILED|TIMEOUT|CANCELLED|NODE_FAIL)
                log_error "Job $job_id failed for ${struct_type}/${analysis_type} with state: $state"
                return 1
                ;;
            RUNNING)
                log_info "Job $job_id still running for ${struct_type}/${analysis_type}..."
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
    local struct_type=$1
    local analysis_type=$2
    local output_dir="results/${struct_type}/${analysis_type}"
    
    if [ ! -f "${output_dir}/distances.npy" ]; then
        log_error "Output file not found: ${output_dir}/distances.npy"
        return 1
    fi
    
    # Check file size
    local size=$(stat -f%z "${output_dir}/distances.npy" 2>/dev/null || stat -c%s "${output_dir}/distances.npy")
    if [ "$size" -eq 0 ]; then
        log_error "Empty output file: ${output_dir}/distances.npy"
        return 1
    }
    
    return 0
}

# Structures to analyze
STRUCTURES=(
    "MoS2_WSe2_1D"
    "MoS2_WSe2_2D"
)

# Types of analysis
TYPES=(
    "interlayer"
    "intralayer"
)

# Launch jobs
active_jobs=()
for STRUCT in "${STRUCTURES[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        log_info "Processing ${STRUCT}/${TYPE}"
        
        # Verify input structure
        if ! verify_input_structure "$STRUCT" "$TYPE"; then
            continue
        fi
        
        # Create output directory
        OUTPUT_DIR="results/${STRUCT}/${TYPE}"
        mkdir -p "$OUTPUT_DIR" || {
            log_error "Failed to create output directory: $OUTPUT_DIR"
            continue
        }
        
        # Launch job
        log_info "Launching job for ${STRUCT}/${TYPE}"
        job_id=$(sbatch --parsable \
            --exclusive -n 1 \
            --cpus-per-task=128 \
            python 3_calculate_distances.py \
            --relaxed-dir "relaxed_structures/${STRUCT}" \
            --reference "structures/${STRUCT}.xyz" \
            --output "${OUTPUT_DIR}/distances.npy")
        
        if [ $? -ne 0 ]; then
            log_error "Failed to launch job for ${STRUCT}/${TYPE}"
            continue
        fi
        
        active_jobs+=("$job_id:$STRUCT:$TYPE")
        log_info "Launched job $job_id"
    done
done

# Monitor jobs
failed_jobs=0
for job_info in "${active_jobs[@]}"; do
    IFS=':' read -r job_id struct_type analysis_type <<< "$job_info"
    
    if ! monitor_job "$job_id" "$struct_type" "$analysis_type"; then
        ((failed_jobs++))
        continue
    fi
    
    # Verify outputs
    if ! verify_outputs "$struct_type" "$analysis_type"; then
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
tar -czf "${LOG_DIR}/distances_${TIMESTAMP}_logs.tar.gz" "$LOG_FILE"
