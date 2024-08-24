# improved-diffusion

This code is hugely based on the codebase of: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage

Install the environment given as .yml file. Sample.sh file runs the code, to make it executable, run chmod +x sample.sh command in the terminal. The accel flag in sample.sh determines whether accelerated or normal stochastic sampling is being run. To change number of sampled images do into scripts/new_image_sample.py and in the end change the number of images and batch size(50,000 and 500 correspondingly for benchmarking). For testing I used 10 for both of these values. All the terminal outputs are saved into a text file in a folder specified in sample.sh. After sampling is done run fidEval.sh for computing FID scores(works for larger amount of images only). In fidEval.sh change the path to the folder in which sampled images are for which you compute the FID score. The FID score is going to be saved in the same directory as images in .txt file inside fid subfolder. 

