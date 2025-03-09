# Object Detection and Model Retraining Application

## Introduction
This repository implements Python code for detecting and labeling objects in a video feed using the YOLO (You Only Look Once) object detection model. The application allows the user to select a bounding box on the video screen and press 'p' to record 10 seconds of cropped bounding box data for YOLO model retraining. After the recording phase, the code will prompt the user to enter a classification for the selected item, person, or animal.

## Key Features
- **Real-Time Object Detection:** Utilizes the YOLOv8 model for object detection from a live video feed.
- **Dynamic Model Retraining:** Allows the user to label detected objects and retrain the model on-the-fly.
- **Custom Object Classification:** Supports adding new custom classifications through video-based data collection.
- **Automatic Data Management:** Manages dataset preparation, including image saving, label generation, and dataset configuration updates.
- **Persistent Model Updates:** Saves the retrained model (`custom_yolov8n.pt`) for future use.

## Installation
1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/object-detection-app.git
cd object-detection-app
```

2. **Install Required Dependencies:**
```bash
pip install -r requirements.txt
```

## Usage
1. **Start the Application:**
```bash
python object_detection_app.py
```

2. **During Video Feed:**
- **Press 'p'** to select a bounding box and start recording data for retraining.
- **Enter a Classification Name** when prompted to label the selected object.

3. **To Quit:**
- **Press 'q'** to close the video feed and exit the application.

## Model Retraining Workflow
1. Select an object in the video feed using the bounding box.
2. Record video data for 10 seconds for the selected object.
3. Provide a custom label for the object when prompted.
4. The application updates the `dataset.yaml` and `labels.json` files.
5. The YOLO model is retrained using the new labeled data.
6. The updated model is saved as in `runs/detect/train/weights`.
7. The application restarts using the retrained model for improved detection.

## Resetting the Environment
To reset the application and clear previous data, run the `autoclean.py` script:
```bash
python autoclean.py
```
This will:
- Clear the `dataset/train` images and labels
- Clear the cache
- Reset `dataset.yaml` and `labels.json` files

## Known Issues
- Ensure the camera is properly connected before starting the application.
- The retraining process may take longer depending on the amount of collected data and system performance.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

