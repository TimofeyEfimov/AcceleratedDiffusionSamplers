import numpy as np

# Replace 'your_file.npy' with the path to your actual file
npy_file_path = '/home/tefimov/dpm-solver/examples/ddpm_and_guided-diffusion/experimentsAccel/cifar10/generalized_singlestep_order1_10_quad_type-dpmsolver/fid.npy'

# Load the array from the .npy file
array = np.load(npy_file_path)

# Now, you want to save this array into a .npz file format
# Replace 'your_compressed_file.npz' with your desired output file path
npz_file_path = '/home/tefimov/dpm-solver/examples/ddpm_and_guided-diffusion/experimentsAccel/cifar10/generalized_singlestep_order1_10_quad_type-dpmsolver/Accel10stepsCifar.npz'

# Save the array to an .npz file (you can add multiple arrays if needed)
np.savez_compressed(npz_file_path, array=array)
