import os
import shutil
import yaml
from pathlib import Path

# Directories for cleaning
images_dir = Path("./dataset/images/train").resolve()
labels_dir = Path("./dataset/labels/train").resolve()
training_data_dir = Path("./training_data").resolve()
runs_dir = Path("./runs").resolve()

# Files to reset
yaml_file = 'dataset.yaml'        # Dataset configuration file
labels_file = 'labels.json'       # File to store custom labels
model_file = 'custom_yolov8n.pt'  # Custom YOLO model file

# Ensure directories exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(training_data_dir, exist_ok=True)

def clear_directory(directory: Path):
    """Remove all files in the specified directory except .gitkeep files."""
    if directory.exists() and directory.is_dir():
        for file in directory.iterdir():
            try:
                if file.is_file() and file.name != '.gitkeep':
                    file.unlink()  # Remove file
                    print(f"🗑️ Deleted file: {file}")
                elif file.is_dir():
                    shutil.rmtree(file)  # Remove directory
                    print(f"🗑️ Deleted directory: {file}")
            except Exception as e:
                print(f"❌ Error deleting {file}: {e}")
    else:
        print(f"Directory not found: {directory}")

def clear_cache():
    """Remove cache files related to YOLO and PyTorch."""
    cache_files = ['dataset.labels.cache', 'dataset.images.cache']
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"🗑️ Cache file removed: {cache_file}")
            except Exception as e:
                print(f"❌ Error removing cache file {cache_file}: {e}")

def clear_runs_directory():
    """Remove the 'runs' directory where model training results are stored."""
    if runs_dir.exists() and runs_dir.is_dir():
        try:
            shutil.rmtree(runs_dir)
            print(f"🗑️ Cleared 'runs' directory.")
        except Exception as e:
            print(f"❌ Error clearing 'runs' directory: {e}")

def reset_dataset_yaml(yaml_file='dataset.yaml'):
    """Reset dataset.yaml to its default state."""
    dataset_config = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/train',
        'names': {}
    }
    try:
        with open(yaml_file, 'w') as f:
            yaml.safe_dump(dataset_config, f)
        print(f"✅ Reset {yaml_file} to default configuration.")
    except Exception as e:
        print(f"❌ Error resetting {yaml_file}: {e}")

def reset_labels_json(labels_file='labels.json'):
    """Reset labels.json to an empty dictionary."""
    try:
        with open(labels_file, 'w') as f:
            f.write("{}")
        print(f"✅ Reset {labels_file} to an empty dictionary.")
    except Exception as e:
        print(f"❌ Error resetting {labels_file}: {e}")

def remove_model_file(model_file='custom_yolov8n.pt'):
    """Remove the custom YOLO model file."""
    if os.path.exists(model_file):
        try:
            os.remove(model_file)
            print(f"🗑️ Model file removed: {model_file}")
        except Exception as e:
            print(f"❌ Error removing model file {model_file}: {e}")
    else:
        print(f"Model file not found: {model_file}")

# Execute cleaning steps
print("🚿 Starting environment cleanup...")

# 1. Clean dataset/train images and labels
clear_directory(images_dir)
clear_directory(labels_dir)
clear_directory(training_data_dir)

# 2. Clean cache
clear_cache()

# 3. Clean runs directory
clear_runs_directory()

# 4. Refresh dataset.yaml
reset_dataset_yaml(yaml_file)

# 5. Refresh labels.json
reset_labels_json(labels_file)

# 6. Remove custom YOLO model file
remove_model_file(model_file)

print("🧼 Environment cleanup completed. You are ready for a fresh start!")
