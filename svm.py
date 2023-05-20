
import numpy as np

from sklearn.svm import LinearSVC

seeds = np.load('out/seeds_20k.npy')
labels = np.load('out/labels_20k.npy', allow_pickle=True)

mapping_age = {
    '0-2': 0,
    '3-9': 0,
    '10-19': 0,
    '20-29': 0,
    '30-39': 0,
    '40-49': 0,
    '50-59': 0,
    '60-69': 1,
    'more than 70': 1
}

mapping_glasses = {
    'glasses': 1,
    'no%20glasses': 0
}

mapping_race = {
    'asian': 0,
    'black': 0,
    'indian': 0,
    'latino hispanic': 0,
    'middle eastern': 1,
    'nan':0,
    'white': 0
}

mapping_hair = {
    "Bald": 1,
    "Black": 0,
    "Blond": 0,
    "Brown": 0,
    "Gray": 0,
}

mapping_beard = {
    "No Beard": 0,
    "Beard": 1,
}

mapping_hat = {
    "No Hat": 0,
    "Hat": 1
}

mapping_emotion = {
    "angry": 0,
    "contempt": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "neutral": 0,
    "sad": 0,
    "surprise": 1
}

sex_label = labels[:, 1]
age_label = np.vectorize(mapping_age.get)(labels[:, 2])
glasses_label = np.vectorize(mapping_glasses.get)(labels[:, 3])
emotion_label_1 = np.vectorize(mapping_emotion.get)(labels[:, 4])
emotion_label_2 = np.vectorize(mapping_emotion.get)(labels[:, 5])
hair_label = np.vectorize(mapping_hair.get)(labels[:, 6])
beard_label = np.vectorize(mapping_beard.get)(labels[:, 7])
hat_label = np.vectorize(mapping_hat.get)(labels[:, 8])
race_label = np.vectorize(mapping_race.get)(labels[:, 9])


current = age_label
svm = LinearSVC(dual=False)
svm.fit(seeds, current)
print(svm.score(seeds, current))

np.save("out/directions/20k/age_old.npy", svm.coef_.ravel())
