DEVICES='6'


##########################

# CIFAR-10 (DDPM checkpoint) example

# data="imagenet64"
# sampleMethod='dpmsolver'
# type="dpmsolver"
# steps="100"
# DIS="logSNR"
# order="2"
# method="singlestep"
# workdir="tryingImageNet/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

# CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12351


data="imagenet64"
sampleMethod='ddpm_noisy'
type="dpmsolver"
steps="1000"
DIS="quad"
order="1"
method="singlestep"
workdir="tryingImageNet/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_type-"$type

CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12351


#CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12352

