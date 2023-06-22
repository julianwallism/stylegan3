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

end_value = 12.5
start_value = -end_value
num_steps = 5
step_size = (end_value - start_value) / (num_steps-1)
print(step_size)

alphas = np.arange(start_value, end_value + step_size, step_size).tolist()
print(alphas)

emotion = "disgust" # "happy" "sad" "surprise" "angry"
# images = ["prueba4.npy"]
vectors = ["out/directions/w_space/emotion_"+ emotion+ "_1.npy", "out/directions/w_space/emotion_"+ emotion+ "_2.npy"]

images = sorted(glob.glob("*.npy"))
# get specific vectors
images = [images[23], images[25], images[31]]
# vectors = sorted(glob.glob("out/directions/w_space/hat.npy"))

new_images = []
for image in images:
    latent = np.load(image)
    new_vectors = []
    for idx, vector in enumerate(vectors):
        direction = np.load(vector)
        aux_vectors = []
        for alpha in alphas:
            if idx == 0:
                 alpha = alpha
            aux_vectors.append(latent + alpha * direction)
        new_images.append(aux_vectors)



if num_steps == 5 and len(vectors) == 1:
    for i, image in tqdm(enumerate(new_images)):
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        plt.subplots_adjust(wspace=0, hspace=0)

        for ax in axs.flat:
            ax.axis('off')

        for idx, vector in enumerate(image):
            w = torch.from_numpy(vector).to(device)
            w = w.repeat(16, 1).unsqueeze(0).to(device)

            img = G.synthesis(w, noise_mode="const")
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            axs[idx].imshow(img)

        plt.savefig(f"resultados/race/race_tiny_{emotion}_{i}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()
elif num_steps == 11 and len(vectors) == 1:
    for i, image in tqdm(enumerate(new_images)): # 4 IMAGENES

        fig, axs = plt.subplots(3, 5, figsize=(20, 12))
        plt.subplots_adjust(wspace=0, hspace=0)

        for ax in axs.flat:
            ax.axis('off')

        pngs = []        
        for idx, vector in enumerate(image):

            w = torch.from_numpy(vector).to(device)
            w = w.repeat(16, 1).unsqueeze(0).to(device)

            img = G.synthesis(w, noise_mode="const")
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            pngs.append(img)        
        
        axs[0, 2].imshow(pngs[5])
        axs[0, 2].set_aspect('equal')
        axs[0, 2].set_adjustable('box')

        value = 4
        for idx in range(5):
            # inverse idx so the 0th image is at the end of the row
            
            axs[1, idx].imshow(pngs[value-idx])
            axs[1, idx].set_aspect('equal')
            axs[1, idx].set_adjustable('box')

        for idx in range(5):
            
            axs[2, idx].imshow(pngs[idx+value+2])
            axs[2, idx].set_aspect('equal')
            axs[2, idx].set_adjustable('box')


        # add a frame like border that includes everything except the first row of the grid

        plt.savefig(f"resultados/race/race_mix_{emotion}_{i}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

elif len(vectors) >1:
    ###########################################################
    for i in range(len(images)): #2 images
        num_images = len(vectors)
        fig, axs = plt.subplots(num_images, 5, figsize=(20, num_images*4))
        plt.subplots_adjust(wspace=0, hspace=0)

        for ax in axs.flat:
            ax.axis('off')

        pngs = []
        for image in new_images[i*num_images:(i+1)*num_images]:        
            for idx, vector in enumerate(image):

                w = torch.from_numpy(vector).to(device)
                w = w.repeat(16, 1).unsqueeze(0).to(device)

                img = G.synthesis(w, noise_mode="const")
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                pngs.append(img)
        print(len(pngs))        
    
        for idx1 in range(num_images):
            for idx2 in range(5):
                axs[idx1, idx2].imshow(pngs[idx1*5+idx2])
                axs[idx1, idx2].set_aspect('equal')
                axs[idx1, idx2].set_adjustable('box')
        

        # add a frame like border that includes everything except the first row of the grid

        plt.savefig(f"resultados/emotion/emotion_tiny_{emotion}_{i}.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()
