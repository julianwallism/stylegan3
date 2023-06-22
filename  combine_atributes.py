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

end_value1 = 12.5
start_value1 = 0
step_size1 = (end_value1 - start_value1) / (num_steps-1)

alphas1 = np.arange(start_value1, end_value1 + step_size1, step_size1).tolist()

end_value2 = 12.5
start_value2 = 0
step_size2 = (end_value2 - start_value2) / (num_steps-1)

alphas2 = np.arange(start_value2, end_value2 + step_size2, step_size2).tolist()
alphas = [alphas1, alphas2]

image = "prueba4.npy"
vectors = ["out/directions/w_space/emotion_happy_1.npy", "out/directions/w_space/glasses.npy"]

latent = np.load(image)
vector1 = np.load(vectors[0])
vector2 = np.load(vectors[1])
edited_vector = []
for i in range(len(alphas[0])):
    aux = latent + alphas[0][i] * vector1 + alphas[1][i] * vector2
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

plt.savefig(f"resultados/combination/glasses_happy.jpg", bbox_inches='tight', pad_inches=0)
plt.close()