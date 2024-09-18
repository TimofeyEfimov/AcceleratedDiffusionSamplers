
DEVICES='1'
data="cifar10"
sampleMethod='ddpm_noisy_accel'
type="dpmsolver"
DIS="quad"
order="1"
method="singlestep"


steps_values=(10)

base_directory="numericalAdjustmentNew"

output_directory="${base_directory}/${sampleMethod}"
mkdir -p "$output_directory"

for steps in "${steps_values[@]}"; do
    workdir="${output_directory}/${data}/${method}_order${order}_${steps}_${DIS}_type-${type}"

    CUDA_VISIBLE_DEVICES="$DEVICES" python main.py --config "${data}.yml" --exp="$workdir" --sample --fid --timesteps="$steps" --eta 0 --ni --skip_type="$DIS" --sample_type="$sampleMethod" --dpm_solver_order="$order" --dpm_solver_method="$method" --dpm_solver_type="$type" --port 12352 >> "${output_directory}/${sampleMethod}_log_${steps}.txt" 2>&1
done

