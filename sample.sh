#!/bin/bash

# Set the CUDA devices that should be visible to the script (e.g., "0" for the first GPU or "0,1" for the first two GPUs)
export CUDA_VISIBLE_DEVICES="5"

# Set model flags, diffusion flags (base), and train flags
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
BASE_DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# ENABLE_ACCEL_FLAG=False 

# For benchmarks 
#imesteps=("5" "10" "12" "15" "20" "25" "35" "50" "75" "100")

# For testing 
timesteps=("75")

for ENABLE_ACCEL_FLAG in True False; do

    if [ "$ENABLE_ACCEL_FLAG" = True ]; then
        ACCEL_FLAG="--accel_flag"
        ACCEL_DIR="Accel"s
    else
        ACCEL_FLAG=""
        ACCEL_DIR="Normal"
    fi

    for step in "${timesteps[@]}"; do

        SAVE_DIR="finalAugustNew/${ACCEL_DIR}DDPM/${step}steps"
        OUTPUT_FILE="finalAugustNew/${ACCEL_DIR}DDPM/${step}steps/outputs.txt"

        # Ensure the output directory exists
        mkdir -p $SAVE_DIR
        
        # Ensure the output file directory exists (it might be different from $SAVE_DIR)
        mkdir -p $(dirname $OUTPUT_FILE)
        
        DIFFUSION_FLAGS="$BASE_DIFFUSION_FLAGS --timestep_respacing $step"

        # Execute the Python script with the provided arguments and redirect output only to the file
        python scripts/new_image_sample.py --model_path /home/tefimov/ddpm_ckpt/imagenet64/imagenet64_uncond_100M_1500K.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ACCEL_FLAG --save_dir $SAVE_DIR > $OUTPUT_FILE 2>&1
    done

done 

