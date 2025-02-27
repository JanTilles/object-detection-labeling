import os
import json
import yaml
import cv2
from pathlib import Path

class DatasetHandler:
    def __init__(self, labels_file='handlers/config/labels.json', yaml_file='handlers/config/dataset.yaml'):
        self.labels_file = labels_file
        self.yaml_file = yaml_file
        self.labels_db = self.load_labels()
        self.update_yaml_file()

    def load_labels(self):
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("Warning: labels.json is empty or corrupted. Initializing a new labels dictionary.")
                    return {}
        else:
            print("No labels.json file found. Starting with an empty labels dictionary.")
            return {}

    def update_yaml_file(self):
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        data['nc'] = len(self.labels_db)
        with open(self.yaml_file, 'w') as file:
            yaml.safe_dump(data, file)

    def update_dataset_yaml(self, new_label):
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as f:
                dataset_config = yaml.safe_load(f) or {}
        else:
            dataset_config = {
                'path': './dataset',
                'train': 'images/train',
                'val': 'images/train',
                'names': {},
                'nc': 0
            }

        if new_label not in dataset_config['names'].values():
            new_class_id = len(dataset_config['names'])
            dataset_config['names'][new_class_id] = new_label

            new_label_id = str(new_class_id)
            self.labels_db[new_label_id] = new_label

            with open(self.labels_file, 'w') as f:
                json.dump(self.labels_db, f, indent=4)
                print(f"Updated labels.json with: {self.labels_db}")

            # Update the nc field
            dataset_config['nc'] = len(dataset_config['names'])

            # Write the updated config back to the yaml file
            with open(self.yaml_file, 'w') as f:
                yaml.safe_dump(dataset_config, f)

            print(f"✅ Added new class '{new_label}' with ID {new_class_id} to {self.yaml_file}.")
        else:
            print(f"Class '{new_label}' already exists in {self.yaml_file}.")

    def move_images_and_generate_labels(self, classification_name, class_id, selected_box, frame, training_data_dir, images_dir, labels_dir):
        image_files = list(training_data_dir.glob("*.jpg"))
        for image_file in image_files:
            img = cv2.imread(str(image_file))
            x1, y1, x2, y2 = map(int, selected_box)
            cropped_img = img[y1:y2, x1:x2]
            destination_image_path = images_dir / image_file.name
            cv2.imwrite(str(destination_image_path), cropped_img)
            print(f"Saved cropped image to: {destination_image_path}")

            label_filename = image_file.stem + ".txt"
            label_path = labels_dir / label_filename

            img_width, img_height = img.shape[1], img.shape[0]
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            bbox_width = (x2 - x1) / img_width
            bbox_height = (y2 - y1) / img_height

            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            bbox_width = min(max(bbox_width, 0), 1)
            bbox_height = min(max(bbox_height, 0), 1)

            label_data = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

            with open(label_path, 'w') as label_file:
                label_file.write(label_data)

            print(f"Generated and normalized label file: {label_path} with data: {label_data}")

            # Delete the processed image file from training_data_dir
            os.remove(image_file)
            print(f"Deleted processed image file: {image_file}")

        print("Cleared training_data folder after processing images and generating labels.")
