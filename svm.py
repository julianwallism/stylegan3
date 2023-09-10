from sklearn.svm import LinearSVC
import numpy as np

seeds = np.load('out/seeds_20k_w.npy')
labels = np.load('out/labels_20k.npy', allow_pickle=True)

# Modify labels for 'age' according to the new classes
labels[:, 2][np.isin(labels[:, 2], ['0-2', '3-9'])] = '0-9'
labels[:, 2][np.isin(labels[:, 2], ['60-69', 'more than 70'])] = '60+'
labels[:, 2][~np.isin(labels[:, 2], ['0-9', '60+'])] = 'other'

# List of all label mappings
label_mappings = [
    {
        'name': 'sex',
        'index': 1,
        'values': ['male', 'female'],
    },
    {
        'name': 'age',
        'index': 2,
        'values': ['0-9', '60+', 'other'],
    },
    {
        'name': 'glasses',
        'index': 3,
        'values': ['glasses', 'no%20glasses'],
    },
    {
        'name': 'emotion_1',
        'index': 4,
        'values': ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    },
    {
        'name': 'emotion_2',
        'index': 5,
        'values': ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    },
    {
        'name': 'hair',
        'index': 6,
        'values': ["Bald", "Black", "Blond", "Brown", "Gray"],
    },
    {
        'name': 'beard',
        'index': 7,
        'values': ["Beard", "No Beard"],
    },
    {
        'name': 'hat',
        'index': 8,
        'values': ["Hat", "No Hat"],
    },
    {
        'name': 'race',
        'index': 9,
        'values': ['asian', 'black', 'indian', 'latino hispanic', 'middle eastern', 'nan', 'white'],
    }
]

for label_mapping in label_mappings:
    name = label_mapping['name']
    values = label_mapping['values']
    index = label_mapping['index']

    # Check if it's a binary label
    if len(values) == 2:
        mapping = {v: 0 for v in values}  # Combine classes into one
        mapping[values[0]] = 1
        print(mapping)

        current_label = np.vectorize(mapping.get)(labels[:, index])

        # Train and evaluate the model
        svm = LinearSVC(dual=False)
        svm.fit(seeds, current_label)
        score = svm.score(seeds, current_label)
        print(f"Score for {name}: {score}")

        # Save the coefficient
        np.save(f"out/PRUEBA/{name}.npy", svm.coef_.ravel())

    else:
        for value in values:
            # Create a deep copy of mapping
            mapping = {v: 0 for v in values}
            mapping[value] = 1

            print(mapping)

            # Prepare labels
            current_label = np.vectorize(mapping.get)(labels[:, index], 0)

            # Train and evaluate the model
            svm = LinearSVC(dual=False)
            svm.fit(seeds, current_label)
            score = svm.score(seeds, current_label)
            print(f"Score for {name} - {value}: {score}")

            # Save the coefficient
            np.save(f"out/PRUEBA/{name}_{value}.npy", svm.coef_.ravel())
