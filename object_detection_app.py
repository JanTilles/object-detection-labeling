from ultralytics import YOLO
import cv2
import os
import time
import json
import yaml
import shutil
from pathlib import Path

# Paths for model and label files
model_path = "runs/detect/train/weights/best.pt"  # Path to the retrained model
default_model_path = "yolo11n.pt"  # Path to the default YOLO model
labels_file = 'labels.json'       # File to store custom labels
yaml_file = 'dataset.yaml'        # Dataset configuration file

# Directories for dataset and training data
training_data_dir = Path("./training_data").resolve()
images_dir = Path("./dataset/images/train").resolve()
labels_dir = Path("./dataset/labels/train").resolve()

# Ensure directories exist
os.makedirs(training_data_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

print(f"Training data directory: {training_data_dir}")
print(f"Images directory: {images_dir}")
print(f"Labels directory: {labels_dir}")

# Initialize labels_db globally
labels_db = {}

# Load or initialize custom labels from labels.json
if os.path.exists(labels_file):
    with open(labels_file, 'r') as f:
        try:
            labels_db = json.load(f)
            print(f"Loaded labels: {labels_db}")
        except json.JSONDecodeError:
            print("Warning: labels.json is empty or corrupted. Initializing a new labels dictionary.")
            labels_db = {}
else:
    print("No labels.json file found. Starting with an empty labels dictionary.")


# Function to update dataset.yaml dynamically
def update_dataset_yaml(new_label, yaml_file='dataset.yaml'):
    global labels_db  # Access the global labels_db variable
    
    # Load existing YAML configuration
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            dataset_config = yaml.safe_load(f) or {}
    else:
        # Create a new configuration if the file doesn't exist
        dataset_config = {
            'path': './dataset',
            'train': 'images/train',
            'val': 'images/train',
            'names': {}
        }

    # Add the new label to the dataset configuration
    if new_label not in dataset_config['names'].values():
        new_class_id = len(dataset_config['names'])
        dataset_config['names'][new_class_id] = new_label

        # Update labels.json with the new label
        new_label_id = str(new_class_id)
        labels_db[new_label_id] = new_label
        
        # Save the updated labels to labels.json
        with open(labels_file, 'w') as f:
            json.dump(labels_db, f, indent=4)
            print(f"Updated labels.json with: {labels_db}")

        # Write the updated configuration back to dataset.yaml
        with open(yaml_file, 'w') as f:
            yaml.safe_dump(dataset_config, f)

        print(f"Updated {yaml_file} with new label: {new_label}")


# Function to move images and generate labels using the selected bounding box
def move_images_and_generate_labels(classification_name: str, class_id: int, selected_box: list, frame):
    """Move images from training_data to dataset and generate labels with the correct bounding box."""
    image_files = list(training_data_dir.glob("*.jpg"))
    for image_file in image_files:
        # Read the image
        img = cv2.imread(str(image_file))
        
        # Crop the image using the selected bounding box
        x1, y1, x2, y2 = map(int, selected_box)  # Convert coordinates to integers
        cropped_img = img[y1:y2, x1:x2]
        
        # Save the cropped image to the dataset folder
        destination_image_path = images_dir / image_file.name
        cv2.imwrite(str(destination_image_path), cropped_img)
        print(f"Saved cropped image to: {destination_image_path}")
        
        # Generate a label file in YOLO format using the selected bounding box
        label_filename = image_file.stem + ".txt"
        label_path = labels_dir / label_filename
        
        # Convert bounding box to YOLO format
        img_width, img_height = img.shape[1], img.shape[0]  # Use original image dimensions
        
        # Calculate YOLO format values
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        bbox_width = (x2 - x1) / img_width
        bbox_height = (y2 - y1) / img_height
        
        # Ensure coordinates are within bounds [0, 1]
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        bbox_width = min(max(bbox_width, 0), 1)
        bbox_height = min(max(bbox_height, 0), 1)
        
        label_data = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
        
        with open(label_path, 'w') as label_file:
            label_file.write(label_data)
        
        print(f"Generated and normalized label file: {label_path} with data: {label_data}")

# Function to clear the training_data directory
def clear_training_data():
    """Remove all files in the training_data directory."""
    for file in training_data_dir.glob("*"):
        try:
            if file.is_file():
                file.unlink()  # Remove file
                print(f"🗑️ Deleted file: {file}")
            elif file.is_dir():
                shutil.rmtree(file)  # Remove directory
                print(f"🗑️ Deleted directory: {file}")
        except Exception as e:
            print(f"❌ Error deleting {file}: {e}")

# Function to retrain the model
def retrain_model(classification_name: str, selected_box: list, frame):
    global model  # Declare model as global at the beginning
    print(f"Retraining model with new label: {classification_name}")
    
    # Load dataset configuration and determine class ID
    with open(yaml_file, 'r') as f:
        dataset_config = yaml.safe_load(f)
        
    class_id = list(dataset_config['names'].keys())[
        list(dataset_config['names'].values()).index(classification_name)
    ]
    
    # Move images and generate labels for retraining
    move_images_and_generate_labels(classification_name, int(class_id), selected_box, frame)
    
    # Clear the training_data directory
    clear_training_data()
    
    # Start model training
    print("Starting model training...")
    model.train(data='dataset.yaml', epochs=20)  # Increase epochs to 20
    
    # Define path for the exported model
    export_path = Path("runs/detect/custom_yolo_model/weights/best.pt")
    
    # Check if the export file exists
    if export_path.exists():
        print(f"✅ Model exported and saved as {export_path}")
        # Load the retrained model
        model = YOLO(export_path)
        print("✅ Retrained model loaded successfully.")
    else:
        print(f"❌ Error: Model export failed. File not found at {export_path}.")

# Validate model file and load the YOLO model
if os.path.exists(model_path) and model_path.endswith('.pt'):
    print(f"Loading custom model from {model_path}")
    try:
        # Validate file size to avoid empty/corrupt models
        if Path(model_path).stat().st_size < 1024:
            raise ValueError("Model file is too small, likely corrupt.")
        
        # Attempt to load the model
        model = YOLO(model_path)
        print("✅ Custom model loaded successfully.")
        
        # Update model names with labels from dataset.yaml
        with open(yaml_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
            model.model.names = dataset_config['names']
            print(f"Model names updated: {model.model.names}")
        
    except Exception as e:
        print(f"❌ Failed to load custom model: {e}")
        print("Loading default YOLOv11 model instead...")
        model = YOLO(default_model_path)
else:
    print("Custom model not found or invalid format. Loading default YOLOv11 model.")
    model = YOLO(default_model_path)


# Initialize the camera
camera = cv2.VideoCapture(0)

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Original 640
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Original 480

# Set additional camera settings for better picture quality
camera.set(cv2.CAP_PROP_BRIGHTNESS, 200)  # Adjust brightness (0-255)
camera.set(cv2.CAP_PROP_CONTRAST, 50)     # Adjust contrast (0-255)
camera.set(cv2.CAP_PROP_SATURATION, 50)   # Adjust saturation (0-255)
camera.set(cv2.CAP_PROP_EXPOSURE, -1)     # Adjust exposure (-1 to -13 for webcams)

recording = False
selected_box = None
start_time = None

confidence_threshold = 0.1  # Lower the confidence threshold

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Run YOLO inference
    results = model(frame)
    print(f"Model predictions: {results}")

    # Draw bounding boxes
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)  # Ensure class ID is an integer
        if score >= confidence_threshold:
            label = model.model.names.get(class_id, "Unknown")
            print(f"Bounding box: {x1, y1, x2, y2}, Score: {score}, Class ID: {class_id}, Label: {label}")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if recording and selected_box is not None:
        x1, y1, x2, y2 = selected_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        img_filename = f"{int(time.time() * 1000)}.jpg"
        img_path = training_data_dir / img_filename
        success = cv2.imwrite(str(img_path), frame)
        if success:
            print(f"✅ Saved image: {img_path}")

    cv2.imshow('YOLO Object Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p') and len(results[0].boxes.data) > 0:
        selected_box = results[0].boxes.data[0][:4]
        print(f"Selected bounding box: {selected_box}")
        recording = True
        start_time = time.time()
    elif recording and (time.time() - start_time > 10):
        recording = False
        classification_name = input("Enter classification name for the selected object: ")
        update_dataset_yaml(classification_name)
        cv2.destroyAllWindows()  # Close the video feed window
        retrain_model(classification_name, selected_box, frame)

camera.release()
cv2.destroyAllWindows()