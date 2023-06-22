# given a folder change pngs to jpgs in the folder

import os
import glob
from PIL import Image
import numpy as np

import torch
import dnnlib
import legacy


device = torch.device('cuda')
with dnnlib.util.open_url("models/ffhq1024.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

save_path = "imagenes/"
# loop through images that end in .npy
for filename in sorted(glob.glob('*.npy')):
    vector = np.load(filename)
    w = torch.from_numpy(vector).to(device)
    w = w.repeat(16, 1).unsqueeze(0).to(device)
    img = G.synthesis(w, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    # save image
    img = Image.fromarray(img, 'RGB')
    img.save(save_path + filename[:-4] + '.jpg')