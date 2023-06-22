import glob
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HUGGINGFACE MODELS
extractor_age = AutoFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")


print("Extractors loaded")

model_age = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
model_age = model_age.to(device)
# models = [model_emotion_1, model_emotion_2]

image_dir = "FID"

# Define transformations for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

batch_size = 50
num_images = 1000

# order listdir
lst = ["thirty", "thirty_inv"]

# Create DataFrame
df = pd.DataFrame()

# Define first column as the name of the images
names = []
for i in range(0, len(lst)):
    names.append("image_" + str(i))

# df['image'] = names

image_names = [f for f in sorted(os.listdir(image_dir+"/original"))]
# print(image_names)


for folder in tqdm(lst):
    predictions = []
    for batch_start in tqdm(range(0, num_images, batch_size)):
        batch_end = batch_start + batch_size
        batch_images = [transform(Image.open(os.path.join(image_dir, folder, img_path)).convert('RGB')) for img_path in image_names[batch_start:batch_end]]
        inputs = extractor_age(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_age(**inputs).logits
            _, preds = torch.max(logits, -1)
            preds = preds.cpu().numpy()
            preds = [model_age.config.id2label[pred.item()] for pred in preds]
            predictions.extend(preds)
    # append preds to df[folder]
    df[folder] = predictions
    #print the length of preds
    print(len(predictions), len(df[folder]))
    

df.to_csv("age2.csv", index=False)

# for folder in tqdm(lst):
#     if os.path.isdir(os.path.join(image_dir, folder)):

#         preds_1 = []
#         preds_2 = []

#         for batch_start in tqdm(range(0, num_images, batch_size)):
#             batch_end = batch_start + batch_size
#             batch_images = [transform(Image.open(os.path.join(image_dir, folder, img_path)).convert('RGB')) for img_path in image_names[batch_start:batch_end]]
            
#             for model, extractor, pred_list in zip(models, extractors, [preds_1, preds_2]):
#                 model = model.to(device)
#                 inputs = extractor(images=batch_images, return_tensors="pt").to(device)
#                 with torch.no_grad():
#                     logits = model(**inputs).logits
#                     _, preds = torch.max(logits, -1)
#                     preds = preds.cpu().numpy()
#                     preds = [model.config.id2label[pred.item()] for pred in preds]
#                     pred_list.extend(preds)

#         df[folder + "_1"] = preds_1
#         df[folder + "_2"] = preds_2

# df.to_csv("test.csv", index=False)
