import os
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml
import json
import cv2  # Assuming OpenCV is used for video processing
import torch  # Assuming PyTorch is used for the model

class YOLOModelHandler:
    def __init__(self, model_path='custom_yolov8n.pt', yaml_file='handlers/config/dataset.yaml'):
        self.model_path = model_path
        self.yaml_file = yaml_file
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path) and self.model_path.endswith('.pt'):
            print(f"Loading custom model from {self.model_path}")
            try:
                if Path(self.model_path).stat().st_size < 1024:
                    raise ValueError("Model file is too small, likely corrupt.")
                model = YOLO(self.model_path)
                print("✅ Custom model loaded successfully.")
                with open(self.yaml_file, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                    model.model.names.update(dataset_config['names'])
                    if len(model.model.names) != model.model.nc:
                        raise ValueError("Number of classes in dataset.yaml does not match the model's output dimensions.")
                    print(f"Model names updated: {model.model.names}")
                return model
            except Exception as e:
                print(f"❌ Failed to load custom model: {e}")
                print("Loading default YOLOv8 model instead...")
                return YOLO("yolov8n.pt")
        else:
            print("Custom model not found or invalid format. Loading default YOLOv8 model.")
            return YOLO("yolov8n.pt")

    def retrain_model(self, classification_name, selected_box, frame, dataset_handler, training_data_dir, images_dir, labels_dir):
        print(f"Retraining model with new label: {classification_name}")
        with open(self.yaml_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
        class_id = list(dataset_config['names'].keys())[
            list(dataset_config['names'].values()).index(classification_name)
        ]
        dataset_handler.move_images_and_generate_labels(classification_name, int(class_id), selected_box, frame, training_data_dir, images_dir, labels_dir)
        print("Starting model training...")
        self.model.train(data='handlers/config/dataset.yaml', epochs=20, name='custom_yolo_model')
        print("Exporting model...")
        self.model.export(format='torchscript')
        export_path = Path("runs/detect/custom_yolo_model/weights/best.torchscript")
        final_model_path = Path("custom_yolov8n.pt")
        wait_time = 0
        while not export_path.exists() and wait_time < 30:
            print("Waiting for model export to complete...")
            time.sleep(1)
            wait_time += 1
        if export_path.exists():
            shutil.copy(export_path, final_model_path)
            print(f"✅ Model exported and saved as {final_model_path}")
        else:
            print(f"❌ Error: Model export failed. File not found at {export_path} after waiting {wait_time} seconds.")
        self.model.save(final_model_path)
        print(f"✅ Model saved correctly as {final_model_path}")
