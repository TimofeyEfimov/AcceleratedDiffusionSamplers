import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    

    dist_util.setup_dist()
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print("sampling...")
    all_images = []
    all_labels = []
    print("Args num samples is:", args.num_samples)
    #accel_Flag = args.accel_Flag 
    accel_flag = args.accel_flag

    print("accel Flag is:", accel_flag)
    # Use the directory to save PNG images from argument
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            accel_flag=accel_flag
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        print('trying to save images')
        for i, single_image in enumerate(sample):
            img = Image.fromarray(single_image.cpu().numpy(), 'RGB')
            img.save(os.path.join(save_dir, f"sample_{len(all_images) * args.batch_size + i}.png"))
        print("successfully saved")
        all_images.append(sample.cpu().numpy())
        if args.class_cond:
            all_labels.append(classes.cpu().numpy())
        
        print(f"Created {len(all_images) * args.batch_size} samples")

    print("Sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=10,
        use_ddim=False,
        model_path="",
        save_dir="sample_images",  # Default save directory
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    add_dict_to_argparser(parser, defaults)
    
    #parser.add_argument('--accel_flag', action='store_true', help='Flag to enable acceleration')
    parser.add_argument('--accel_flag', action='store_true', help='Flag to enable acceleration')

    return parser

if __name__ == "__main__":  
    main()
