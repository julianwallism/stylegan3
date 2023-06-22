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
extractor_emotion_1 = AutoFeatureExtractor.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
extractor_emotion_2 = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")
extractors = [extractor_emotion_1, extractor_emotion_2]

print("Extractors loaded")

model_emotion_1 = AutoModelForImageClassification.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
model_emotion_2 = AutoModelForImageClassification.from_pretrained("Rajaram1996/FacialEmoRecog")
models = [model_emotion_1, model_emotion_2]

image_dir = "FID"

# Define transformations for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define a function to perform prediction on a single image
def predict_image(model, extractor, img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = extractor(img_tensor)
        outputs = model(**features)
        _, preds = torch.max(outputs.logits, 1)
    return preds.item()

# Define the batch processing function
def batch_processing(models, extractors, img_paths, batch_size=50):
    predictions = []
    for model, extractor in zip(models, extractors):
        model = model.to(device)
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            batch_preds = [predict_image(model, extractor, img_path) for img_path in batch_paths]
            predictions.extend(batch_preds)
    return predictions

# Get the subfolder names
subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]

# Initialize the dataframe
df = pd.DataFrame()

# Iterate over the subfolders and files within
for subfolder in tqdm(subfolders):
    img_paths = sorted(glob.glob(os.path.join(subfolder, '*.png')))
    for model_idx, (model, extractor) in enumerate(zip(models, extractors)):
        preds = batch_processing([model], [extractor], img_paths)
        df[f'{os.path.basename(subfolder)}_model_{model_idx+1}'] = preds

print(df.head())
