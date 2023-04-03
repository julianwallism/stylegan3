import numpy as np
from sklearn.linear_model import LinearRegression

seeds = np.load("out/seeds/seeds.npy")
labels = np.load("out/labels.npy")

# get rid of the header row
labels = labels[1:]



# DeepFace_Age is index 1
# ViT_AgeClass is index 5
# ViT_Age is index 7
y1 = labels[:, 1].astype(float)
y2 = labels[:, 5].astype(float)
y3 = labels[:, 7].astype(float)


print(seeds.shape)

# Reshape seeds to a 2D array with shape (n_samples, n_features)
n_samples = seeds.shape[0]
n_features = seeds.shape[1] * seeds.shape[2]
seeds_2d = seeds.reshape((n_samples, n_features))

print(seeds_2d.shape)

reg2 = LinearRegression().fit(seeds_2d, y2)
print(reg2.score(seeds_2d, y2))

reg1 = LinearRegression().fit(seeds_2d, y1)
print(reg1.score(seeds_2d, y1))

reg3 = LinearRegression().fit(seeds_2d, y3)
print(reg3.score(seeds_2d, y3))