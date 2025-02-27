import time
import cv2
from pathlib import Path
from handlers.dataset_handler import DatasetHandler
from handlers.yolo_model_handler import YOLOModelHandler
from handlers.camera_handler import CameraHandler

# Initialize handlers
dataset_handler = DatasetHandler()
yolo_model_handler = YOLOModelHandler()
camera_handler = CameraHandler()

training_data_dir = Path("./training_data").resolve()
images_dir = Path("./dataset/images/train").resolve()
labels_dir = Path("./dataset/labels/train").resolve()

recording = False
selected_box = None
start_time = None
confidence_threshold = 0.1

while True:
    frame = camera_handler.read_frame()
    if frame is None:
        break

    results = yolo_model_handler.model(frame)
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)
        if score >= confidence_threshold:
            label = yolo_model_handler.model.model.names.get(class_id, "Unknown")
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
        recording = True
        start_time = time.time()
    elif recording and (time.time() - start_time > 10):
        recording = False
        classification_name = input("Enter classification name for the selected object: ")
        dataset_handler.update_dataset_yaml(classification_name)
        cv2.destroyAllWindows()
        yolo_model_handler.retrain_model(classification_name, selected_box, frame, dataset_handler, training_data_dir, images_dir, labels_dir)

camera_handler.release()
