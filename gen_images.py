# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default="models/ffhq1024.pkl", required=False)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default=0, required=False)
@click.option('--w_space', is_flag=True, help='Interpolate in W space instead of Z space')
@click.option('--outdir_img', help='Where to save the output images', type=str, default="out/images", required=False, metavar='DIR')
@click.option('--outdir_seeds', help='Where to save the latent vector', type=str, default="out/", required=False, metavar='DIR')
@click.option('--append_latent', is_flag=True, help='If true, append the latent vector to the latent vector file (if it exists). Otherwise, overwrite the file.')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    outdir_img: str,
    outdir_seeds: str,
    w_space: bool, 
    append_latent: bool
    ):
 
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        print("Loading Generator...")
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir_img, exist_ok=True)
    os.makedirs(outdir_seeds, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    vector_seeds = []
    # Generate images.
    for seed in tqdm(seeds):

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        if w_space:
            w = G.mapping(z, None)
            w = w[0][0].unsqueeze(0).cpu().numpy()
            vector_seeds.append(w)
        else:
            vector_seeds.append(z.cpu().numpy())

        img = G(z, label, truncation_psi=1, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir_img}/seed{seed:04d}.png')

    
    vector_seeds = np.concatenate(vector_seeds, axis=0)
    if(append_latent):
        print("Appending latent vector to file...")
        if(os.path.exists(f'{outdir_seeds}/seeds.npy')):
            old_vector_seeds = np.load(f'{outdir_seeds}/seeds.npy')
            vector_seeds = np.concatenate((old_vector_seeds, vector_seeds), axis=0)
    np.save(f'{outdir_seeds}/seeds.npy', vector_seeds)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()

#----------------------------------------------------------------------------
