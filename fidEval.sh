
export CUDA_VISIBLE_DEVICES="1"

# Set model flags, diffusion flags (base), and train flags
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
BASE_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# For benchmarks 
#imesteps=("5" "10" "12" "15" "20" "25" "35" "50" "75" "100")

# For testing 
timesteps=("75")

# Pre computed statistics for Image Net 64 

FID_STATS_FILE="/home/tefimov/dpm-solver/examples/ddpm_and_guided-diffusion/fid_stats/fid_stats_imagenet64_train.npz"

# Now, iterate over the same timestep directories to calculate FID scores
for step in "${timesteps[@]}"; do
    # Define the directory and output file based on the timestep
    SAVE_DIR="finalAugustNew/AccelDDPM//${step}steps"
    FID_DIR="${SAVE_DIR}/fid" 
    FID_OUTPUT_FILE="${FID_DIR}/fid_results.txt"

    # Ensure the FID directory exists
    mkdir -p $FID_DIR

    # Run the PyTorch FID script for each directory and save its output
    echo "Calculating FID for ${step} steps..."
    python -m pytorch_fid ${SAVE_DIR} $FID_STATS_FILE > $FID_OUTPUT_FILE 2>&1
    echo "FID calculation complete for ${step} steps, output saved to ${FID_OUTPUT_FILE}"
done
