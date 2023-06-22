import numpy as np

import torch
import dnnlib
import legacy
import matplotlib.pyplot as plt
import glob
from matplotlib.lines import Line2D

from tqdm import tqdm

device = torch.device('cuda')
with dnnlib.util.open_url("models/ffhq1024.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)


# alphas = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
num_steps = 5
start_value = 0

end_value1 = -7
step_size1 = (end_value1 - start_value) / (num_steps-1)

alphas1 = np.arange(start_value, end_value1 + step_size1, step_size1).tolist()

end_value2 = 5
step_size2 = (end_value2 - start_value) / (num_steps-1)

alphas2 = np.arange(start_value, end_value2 + step_size2, step_size2).tolist()

end_value3 = 2.5
step_size3 = (end_value3 - start_value) / (num_steps-1)

alphas3 = np.arange(start_value, end_value3 + step_size3, step_size3).tolist()

alphas = [alphas1, alphas2, alphas3]

image = "bald_4.npy"
vectors = ["out/directions/w_space/hair_bald.npy", "out/directions/w_space/beard.npy", "out/directions/w_space/race_black.npy"]


latent = np.load(image)
vector1 = np.load(vectors[0])
vector2 = np.load(vectors[1])
vector3 = np.load(vectors[2])
edited_vector = []
for i in range(len(alphas[0])):
    aux = latent + alphas[0][i] * vector1 + alphas[1][i] * vector2 + alphas[2][i] * vector3
    edited_vector.append(aux)


fig, axs = plt.subplots(1, 5, figsize=(20, 4))
plt.subplots_adjust(wspace=0, hspace=0)

for ax in axs.flat:
    ax.axis('off')

for idx, vector in enumerate(edited_vector):
    w = torch.from_numpy(vector).to(device)
    w = w.repeat(16, 1).unsqueeze(0).to(device)

    img = G.synthesis(w, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    axs[idx].imshow(img)

plt.savefig(f"resultados/combination/hair_beard_black.jpg", bbox_inches='tight', pad_inches=0)
plt.close()