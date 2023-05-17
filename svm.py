
import numpy as np

from sklearn.svm import LinearSVC

seeds = np.load('out/seeds.npy')
labels = np.load('out/good_labels.npy')

mapping_age = {
    '0-2': 0,
    '3-9': 1,
    '10-19': 2,
    '20-29': 3,
    '30-39': 4,
    '40-49': 5,
    '50-59': 6,
    '60-69': 7,
    'more than 70': 8
}

mapping_glasses = {
    'glasses': 1,
    'no%20glasses': 0
}

mapping_race = {
    'asian': 1,
    'black': 0,
    'indian': 0,
    'latino hispanic': 0,
    'middle eastern': 0,
    'nan':0,
    'white': 0
}

mapping_hair = {
    "Bald": 0,
    "Black": 0,
    "Blond": 0,
    "Brown": 0,
    "Gray": 1,
}

mapping_beard = {
    "No Beard": 0,
    "Beard": 1,
}

mapping_hat = {
    "No Hat": 0,
    "Hat": 1
}

mapping_emotion_1 = {
    "angry": 0,
    "happy": 0,
    "neutral": 0,
    "sad": 1
}

mapping_emotion_2 = {
    "anger": 0,
    "contempt": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "neutral": 0,
    "sadness": 0,
    "surprise": 1
}

sex_label = labels[:, 1]
age_label = np.vectorize(mapping_age.get)(labels[:, 2])
glasses_label = np.vectorize(mapping_glasses.get)(labels[:, 3])
emotion_label_1 = np.vectorize(mapping_emotion_1.get)(labels[:, 4])
emotion_label_2 = np.vectorize(mapping_emotion_2.get)(labels[:, 5])
hair_label = np.vectorize(mapping_hair.get)(labels[:, 6])
beard_label = np.vectorize(mapping_beard.get)(labels[:, 7])
hat_label = np.vectorize(mapping_hat.get)(labels[:, 8])
race_label = np.vectorize(mapping_race.get)(labels[:, 9])


svm = LinearSVC(dual=False, class_weight='balanced')
svm.fit(seeds, emotion_label_2)
print(svm.score(seeds, emotion_label_2))

np.save("out/directions/emotion_surprise_svm.npy", svm.coef_.ravel())
