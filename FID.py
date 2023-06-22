import numpy as np
import torch
from torchvision.models import inception_v3
from pytorch_fid import fid_score

import dnnlib
import legacy

from torchvision import transforms
from PIL import Image

from tqdm import tqdm

print("Loading Inception model...")
# Load InceptionV3 model
inception_model = inception_v3(transform_input=False).eval()

print("Loading StyleGAN model...")
device = torch.device('cuda')
with dnnlib.util.open_url("models/ffhq1024.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


print("Loading Latent Vectors...")
# Load the latent vectors from seeds.npy
latent_vectors = np.load('out/seeds_20k.npy')

# Apply vector X
X = np.load("out/directions/20k/sex.npy")
X = np.divide(X, 40)
X = np.multiply(X, 50)
modified_latent_vectors = latent_vectors + X



# Calculate FID without saving images
def calculate_features(latent_vectors):
    features = []
    label = torch.zeros([1, G.c_dim], device=device)
     # Preprocess the modified image for InceptionV3
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for lv in tqdm(latent_vectors):
        # reshape latent vector
        lv = np.reshape(lv, (1, 512))
        lv = torch.from_numpy(lv).to(device)
        # Generate image from modified latent vector using StyleGAN3
        modified_image = G(lv, label, truncation_psi=1, noise_mode='const')
        modified_image = (modified_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        modified_image = Image.fromarray(modified_image, 'RGB')

        modified_image = preprocess(modified_image).unsqueeze(0)

        # Extract features using InceptionV3
        features.append(inception_model(modified_image))

    return torch.cat(features, dim=0)

# Calculate features for the original and modified datasets
original_features = calculate_features(latent_vectors)
modified_features = calculate_features(modified_latent_vectors)

# Calculate FID
fid = fid_score.calculate_frechet_distance(
    original_features, modified_features
)

print("FID:", fid)
