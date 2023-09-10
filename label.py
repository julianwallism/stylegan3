import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from transformers import AutoFeatureExtractor, AutoModelForImageClassification


""" Script to label syntetic images using pretrained models """


#########################################################################
# # HUGGINGFACE MODELS
extractor_gender = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification-2")
extractor_age = AutoFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")
extractor_eyeglasses = AutoFeatureExtractor.from_pretrained("youngp5/eyeglasses_detection")
extractor_emotion_1 = AutoFeatureExtractor.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
extractor_emotion_2 = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")
extractors = [extractor_gender, extractor_age, extractor_eyeglasses, extractor_emotion_1, extractor_emotion_2]

print("Extractors loaded")

model_gender = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
model_age = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
model_eyeglasses = AutoModelForImageClassification.from_pretrained("youngp5/eyeglasses_detection")
model_emotion_1 = AutoModelForImageClassification.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
model_emotion_2 = AutoModelForImageClassification.from_pretrained("Rajaram1996/FacialEmoRecog")
models = [model_gender, model_age, model_eyeglasses, model_emotion_2, model_emotion_2]

print("Hugging Face models loaded")

#########################################################################
# EfficientNet Transfer Learning
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HAIR
checkpoint_path_hair = 'models/hair_ckpt.pt'

model_hair = efficientnet_v2_s(pretrained=True, num_classes=1000)
num_classes = 5  # Assuming you have 5 hair color classes
model_hair.classifier[1] = nn.Linear(model_hair.classifier[1].in_features, num_classes)
model_hair.load_state_dict(torch.load(checkpoint_path_hair))
model_hair = model_hair.to(device)
model_hair.eval()

# Define the label encoder
label_encoder_hair = ["Bald", "Black", "Blond", "Brown", "Gray"]

# BEARD
checkpoint_path_beard = 'models/beard_ckpt.pt'

model_beard = efficientnet_v2_s(pretrained=True, num_classes=1000)
num_classes = 2  
model_beard.classifier[1] = nn.Linear(model_beard.classifier[1].in_features, num_classes)
model_beard.load_state_dict(torch.load(checkpoint_path_beard))
model_beard = model_beard.to(device)
model_beard.eval()

label_encoder_beard = ["No Beard", "Beard"]

# HAT
checkpoint_path_hat = 'models/hat_ckpt.pt'

model_hat = efficientnet_v2_s(pretrained=True, num_classes=1000)
num_classes = 2  
model_hat.classifier[1] = nn.Linear(model_hat.classifier[1].in_features, num_classes)
model_hat.load_state_dict(torch.load(checkpoint_path_hat))
model_hat = model_hat.to(device)
model_hat.eval()

label_encoder_hat = ["No Hat", "Hat"]

print("Transfer Learning models loaded")

#########################################################################

labels = []

# loop through all images in out/images and pass them through the model
image_dir = "out/images"

batch_size = 100

num_images = 20000

for batch_start in tqdm(range(0, num_images, batch_size)):
    batch_end = min(batch_start + batch_size, num_images)
    batch_size_actual = batch_end - batch_start

    batch_images = torch.zeros((batch_size_actual, 3, 224, 224)).to(device)

      # Load and preprocess images in the batch
    for i, idx in enumerate(range(batch_start, batch_end)):
        filename = "seed" + str(idx) + ".png"
        image_path = os.path.join(image_dir, filename)
        img = Image.open(image_path).convert("RGB")
        img = data_transforms(img)
        batch_images[i] = img

        if idx >= len(labels):
            labels = np.vstack([labels, np.zeros(10, dtype=object)])
            labels[idx][0] = filename

    # Pass images through Hugging Face models
    for i in range(len(extractors)):
        extractor = extractors[i]
        model = models[i]
        model = model.to(device)

        inputs = extractor(images=batch_images, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted_class_indices = torch.max(logits, -1)
            predicted_class_indices = predicted_class_indices.cpu().numpy()

        # Update labels for the batch
        for j in range(batch_size_actual):
            labels[batch_start + j][i+1] = model.config.id2label[predicted_class_indices[j]]

    # Pass images through EfficientNet models
    batch_images = batch_images.to(device)
    with torch.no_grad():
        # Hair
        output_hair = model_hair(batch_images)
        _, predicted_hair = torch.max(output_hair, 1)
        predicted_hair = predicted_hair.cpu().numpy()

        # Beard
        output_beard = model_beard(batch_images)
        _, predicted_beard = torch.max(output_beard, 1)
        predicted_beard = predicted_beard.cpu().numpy()

        # Hat
        output_hat = model_hat(batch_images)
        _, predicted_hat = torch.max(output_hat, 1)
        predicted_hat = predicted_hat.cpu().numpy()

    # Update labels for the batch
    for j in range(batch_size_actual):
        labels[batch_start + j][6] = label_encoder_hair[predicted_hair[j]]
        labels[batch_start + j][7] = label_encoder_beard[predicted_beard[j]]
        labels[batch_start + j][8] = label_encoder_hat[predicted_hat[j]]

    # Update labels for the race column
    for j in range(batch_size_actual):
        labels[batch_start + j][9] = "nan"

np.save("out/labels.npy", labels)