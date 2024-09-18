
DEVICES='1'
data="cifar10"

# Choose generalized for vanilla ODE sampler, generailized_accel for accelerate ODE sampler, 
# and ddpm_noisy for vanilla SDE sampler, ddpm_noisy_accel for accelerate SDE sampler

sampleMethod='generalized_accel'
DIS="quad"

steps_values=(10)

base_directory="numericalAdjustmentTest"

output_directory="${base_directory}/${sampleMethod}"
mkdir -p "$output_directory"

for steps in "${steps_values[@]}"; do
    workdir="${output_directory}/${data}"

    CUDA_VISIBLE_DEVICES="$DEVICES" python main.py --config "${data}.yml" --exp="$workdir" --sample --fid --timesteps="$steps" --eta 0 --ni --skip_type="$DIS" --sample_type="$sampleMethod" --port 12352 >> "${output_directory}/${sampleMethod}_log_${steps}.txt" 2>&1
done

