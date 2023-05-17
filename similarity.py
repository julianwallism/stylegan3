import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

directory = "out/directions"

vectors = []
file_names = []  # Store the file names

for filename in sorted(os.listdir(directory)):
    if filename.endswith(".npy"):
        file_path = os.path.join(directory, filename)
        vector = np.load(file_path)
        vector = np.reshape(vector, (512,))
        vectors.append(vector)
        file_names.append(filename[:-4]) 

vectors = np.array(vectors)
similarity_matrix = cosine_similarity(vectors)

plt.imshow(similarity_matrix, cmap = "coolwarm",interpolation='nearest')
plt.colorbar()
plt.title('Similarity Matrix')

# Set the axis labels using file names
plt.xticks(range(len(file_names)), file_names, rotation=90)
plt.yticks(range(len(file_names)), file_names)

# plt.show()

# Save the figure
plt.savefig("out/similarity_matrix.png", bbox_inches='tight')
