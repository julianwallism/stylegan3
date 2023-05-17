import numpy as np

label = np.load("out/good_labels.npy")


unique, counts = np.unique(label[:, 4], return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(label[:, 5], return_counts=True)
print(dict(zip(unique, counts)))
