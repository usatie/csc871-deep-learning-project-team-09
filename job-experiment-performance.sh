#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m3930
#SBATCH --time=04:00:00
#SBATCH --job-name=transformer-perf-exp
#SBATCH --output=log/performance-experiment-%j.out
#SBATCH --error=log/performance-experiment-%j.err

# Load required modules
module load python
conda activate transformer-translation

# Create log directory if it doesn't exist
mkdir -p log/performance_results

# Define experiment parameters
BATCH_SIZES=(16 32 64 128 256 512 1024)
GPU_COUNTS=(1 2 4)
DATASET="tatoeba_zh_en"
NUM_EPOCHS=1  # Using fewer epochs for performance testing
WARMUP=3000

# Function to determine appropriate accumulation iterations based on batch size
# This helps maintain a similar total batch size across different batch_size settings
get_accum_iter() {
    local batch_size=$1
    
    # Scale accumulation iterations inversely with batch size
    # This helps keep the effective batch size similar
    case $batch_size in
        16)  echo 64 ;;
        32)  echo 32 ;;
        64)  echo 16 ;;
        128) echo 8 ;;
        256) echo 4 ;;
        512) echo 2 ;;
        1024) echo 1 ;;
        *) echo 1 ;;
    esac
}

# Main experiment loop
for num_gpus in "${GPU_COUNTS[@]}"; do
    # Set nodes based on GPU count
    if [ "$num_gpus" -eq 1 ]; then
        NODES=1
    elif [ "$num_gpus" -eq 2 ]; then
        NODES=1
    elif [ "$num_gpus" -eq 4 ]; then
        NODES=1
    else
        echo "Invalid GPU count: $num_gpus"
        continue
    fi
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        accum_iter=$(get_accum_iter $batch_size)
        
        echo "Running experiment with $num_gpus GPU(s), batch_size=$batch_size, accum_iter=$accum_iter"
        
        # Create a new job for this specific configuration
        cat << EOF > job-perf-bs${batch_size}-gpu${num_gpus}.sh
#!/bin/bash -l
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=${NODES}
#SBATCH --gpus=${num_gpus}
#SBATCH --account=m3930
#SBATCH --time=01:00:00
#SBATCH --job-name=perf-bs${batch_size}-gpu${num_gpus}
#SBATCH --output=log/perf-bs${batch_size}-gpu${num_gpus}.o%j
#SBATCH --error=log/perf-bs${batch_size}-gpu${num_gpus}.e%j

module load python
conda activate transformer-translation

# Record start time
START_TIME=\$(date +%s)

# Run the training script with performance measurement
python src/scripts/train.py --dataset ${DATASET} --batch-size ${batch_size} --accum-iter ${accum_iter} --warmup ${WARMUP} --num-epochs ${NUM_EPOCHS} --force

# Record end time
END_TIME=\$(date +%s)
ELAPSED_TIME=\$((END_TIME - START_TIME))

# Save results to a structured CSV file
echo "${batch_size},${num_gpus},${accum_iter},\${ELAPSED_TIME}" >> log/performance_results/results.csv

EOF
        
        # Submit the job
        sbatch job-perf-bs${batch_size}-gpu${num_gpus}.sh
        
        # Small delay between submissions
        sleep 1
    done
done

# Create a header for the results file if it doesn't exist
if [ ! -f log/performance_results/results.csv ]; then
    echo "batch_size,num_gpus,accum_iter,elapsed_time_seconds" > log/performance_results/results.csv
fi

echo "All performance experiment jobs submitted!" 