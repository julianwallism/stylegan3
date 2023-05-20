import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

directory = ["10k", "20k"]

for dir in directory:
    vectors = []
    file_names = []  # Store the file names
    # similarity_matrix = np.zeros((len(os.listdir(dir)), len(os.listdir(dir))))

    for filename in sorted(os.listdir("out/directions/"+dir)):
        if filename.endswith(".npy"):
            file_path = os.path.join("out/directions/"+dir, filename)
            vector = np.load(file_path)
            vector = np.reshape(vector, (512,))
            vectors.append(vector)
            file_names.append(filename[:-4]) 

    vectors = np.array(vectors)
    similarity_matrix = cosine_similarity(vectors)
    print(type(similarity_matrix))

    plt.imshow(similarity_matrix, cmap = "coolwarm",interpolation='nearest')
    plt.colorbar()

    # Set the axis labels using file names
    plt.xticks(range(len(file_names)), file_names, rotation=90)
    plt.yticks(range(len(file_names)), file_names)

    # plt.show()

    # Save the figure
    plt.savefig("out/"+dir+"_similarity_matrix.png", bbox_inches='tight')
    plt.clf()

# Combine all files and create similarity matrix
all_vectors = []
all_file_names = []

# Load vectors from both directories
for dir in directory:
    for filename in sorted(os.listdir("out/directions/"+dir)):
        if filename.endswith(".npy"):
            file_path = os.path.join("out/directions/"+dir, filename)
            vector = np.load(file_path)
            vector = np.reshape(vector, (512,))
            all_vectors.append(vector)
            all_file_names.append(filename[:-4]+"_"+dir)

all_vectors = np.array(all_vectors)
all_similarity_matrix = cosine_similarity(all_vectors)

# Sort file names alphabetically
sorted_indices = np.argsort(all_file_names)
sorted_file_names = [all_file_names[i] for i in sorted_indices]
sorted_similarity_matrix = all_similarity_matrix[sorted_indices][:, sorted_indices]

# Increase the figure size
plt.figure(figsize=(12, 10))

plt.imshow(sorted_similarity_matrix, cmap="coolwarm", interpolation='nearest')
plt.colorbar()

plt.xticks(range(len(sorted_file_names)), sorted_file_names, rotation=90)
plt.yticks(range(len(sorted_file_names)), sorted_file_names)

plt.savefig("out/all_similarity_matrix.png", bbox_inches='tight')
plt.clf()