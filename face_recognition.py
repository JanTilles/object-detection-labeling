import os
import cv2
import json
import yaml
import numpy as np
import pickle
from face_recognition import face_encodings, compare_faces, face_distance

# Define paths
IMAGE_PATH = './dataset/images/train'
LABEL_PATH = './dataset/labels/train'
LABELS_JSON = 'labels.json'
DATASET_YAML = 'dataset.yaml'
OUTPUT_DB = 'face_recognition_db.pkl'

# Load label mapping from labels.json and dataset.yaml
with open(LABELS_JSON, 'r') as f:
    labels_dict = json.load(f)

with open(DATASET_YAML, 'r') as f:
    dataset_info = yaml.safe_load(f)

# Inverse mapping of label ids to names
label_map = {int(k): v for k, v in labels_dict.items()}

# Initialize face recognition database
face_db = []

# Iterate over images and labels
for label_file in os.listdir(LABEL_PATH):
    if not label_file.endswith('.txt'):
        continue
    image_file = os.path.join(IMAGE_PATH, label_file.replace('.txt', '.jpg'))
    label_file_path = os.path.join(LABEL_PATH, label_file)
    if not os.path.exists(image_file):
        continue
    
    # Read the image
    image = cv2.imread(image_file)
    if image is None:
        continue

    # Read the labels
    with open(label_file_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:]))
        if class_id not in label_map:
            continue

        # Convert YOLO bbox to actual image coordinates
        h, w, _ = image.shape
        x_center, y_center, box_w, box_h = bbox
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # Extract face from the image
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Get face encodings
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_encodings(rgb_face)
        if len(encodings) == 0:
            continue
        encoding = encodings[0]

        # Store in the face database
        face_db.append({'name': label_map[class_id], 'encoding': encoding})

# Save the face recognition database
with open(OUTPUT_DB, 'wb') as f:
    pickle.dump(face_db, f)

print(f'Face recognition database saved to {OUTPUT_DB}')}
