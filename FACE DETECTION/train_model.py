import cv2
import numpy as np
import os
import pickle

data_dir = "data"
faces = []
labels = []
label_id = 0
label_dict = {}

for person in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person)
    if not os.path.isdir(person_path):
        continue

    label_dict[label_id] = person
    for img_name in os.listdir(person_path):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(label_id)

    label_id += 1

# Train model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")

# Save label dictionary
with open("labels.pickle", "wb") as f:
    pickle.dump(label_dict, f)

print("[INFO] Model trained and saved as 'trained_model.yml'")
