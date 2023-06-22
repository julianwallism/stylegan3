import torch
import os
import dnnlib
import legacy
import numpy as np
import concurrent.futures

from PIL import Image
from tqdm import tqdm

network_pkl = "models/ffhq1024.pkl"
device = torch.device('cuda')

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

def gen_image(w):
    # Generate image from modified latent vector using StyleGAN3
    modified_image = G.synthesis(w, noise_mode='const')
    modified_image = (modified_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    modified_image = Image.fromarray(modified_image, 'RGB')
    return modified_image


def save_image(image, folder, index):
    image_path = os.path.join(folder, f"image_{index}.png")
    image.save(image_path)


batch_size = 50
num_images = 1000
num_batches = num_images // batch_size
og_latent_vectors = np.load("FID/seeds.npy")

name = ["thirty"]

vectors = []
folders = []
for name in name:
    direction_vectors = np.load("out/directions/w_space/age_"+name+".npy")
    vector1 = np.multiply(direction_vectors, 2.5)
    vector2 = np.multiply(direction_vectors, -2.5)
    vectors.append(vector1)
    vectors.append(vector2)

    
    folder1 = "FID/"+name
    folder2 = "FID/"+name+"_inv"
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    folders.append(folder1)
    folders.append(folder2)



# Initialize a ThreadPoolExecutor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

for foldername, vector in tqdm(enumerate(vectors)):
    latent_vectors = og_latent_vectors + vector
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        images = []
        for j in range(start_idx, end_idx):
            w = np.reshape(latent_vectors[j], (1, 16, 512))
            w = torch.from_numpy(w).to(device)
            images.append(gen_image(w))
        # Use the executor to map the save_image function to the images
        list(executor.map(save_image, images, [folders[foldername]]*len(images), range(start_idx, end_idx)))
