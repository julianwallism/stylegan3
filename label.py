import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# from deepface import DeepFace

#########################################################################
# # HUGGINGFACE MODELS
extractor_gender = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification-2")
extractor_age = AutoFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")
extractor_eyeglasses = AutoFeatureExtractor.from_pretrained("youngp5/eyeglasses_detection")
extractor_emotion_1 = AutoFeatureExtractor.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
extractor_emotion_2 = AutoFeatureExtractor.from_pretrained("Rajaram1996/FacialEmoRecog")
extractors = [extractor_gender, extractor_age, extractor_eyeglasses, extractor_emotion_1, extractor_emotion_2]


model_gender = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
model_age = AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
model_eyeglasses = AutoModelForImageClassification.from_pretrained("youngp5/eyeglasses_detection")
model_emotion_2 = AutoModelForImageClassification.from_pretrained("jayanta/google-vit-base-patch16-224-cartoon-emotion-detection")
model_emotion_2 = AutoModelForImageClassification.from_pretrained("Rajaram1996/FacialEmoRecog")
models = [model_gender, model_age, model_eyeglasses, model_emotion_2, model_emotion_2]

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
num_classes = 5  # Assuming you have 4 hair color classes
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

#########################################################################

# labels = []
labels = np.load("out/good_labels.npy").tolist()

# loop through all images in out/images and pass them through the model
image_dir = "out/images"
i = 0
lst = sorted(os.listdir(image_dir))
for idx, filename in tqdm(enumerate(lst)):
    image_path = os.path.join(image_dir, filename)
    img = Image.open(image_path)

    labels.append([filename])

    # HUGGINGFACE MODELS
    for i in range(len(extractors)):
        extractor = extractors[i]
        model = models[i]

        inputs = extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_label = model.config.id2label[predicted_class_idx]
        labels[idx].append(predicted_class_label)

    
    # EFFICIENTNET
    img = img.convert('RGB')
    img = data_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # HAIR
        output = model_hair(img)
        _, predicted = torch.max(output.data, 1)
        label = label_encoder_hair[predicted.item()]
        labels[idx].append(label)

        # BEARD
        output = model_beard(img)
        _, predicted = torch.max(output.data, 1)
        label = label_encoder_beard[predicted.item()]
        labels[idx].append(label)

        # HAT
        output = model_hat(img)
        _, predicted = torch.max(output.data, 1)
        label = label_encoder_hat[predicted.item()]
        labels[idx].append(label)

    # # DEEPFACE
    # objs = DeepFace.analyze(img_path=image_path, actions=['race'], silent=True)
    # race = objs[0]['dominant_race']
    # labels[idx].append(race)


np.save("out/good_labels.npy", labels)



