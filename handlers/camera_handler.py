import cv2

class CameraHandler:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.set_camera_properties()

    def set_camera_properties(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 150)
        self.camera.set(cv2.CAP_PROP_CONTRAST, 50)
        self.camera.set(cv2.CAP_PROP_SATURATION, 50)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, -4)

    def read_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return None
        return frame

    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()
