import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

# import pathlib
# For Windows
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

def load_models():
    detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "Face", (0, 0))
    
    model = tf.keras.models.load_model("models/model.h5")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=5)
    
    return detector, model, face_mesh


# Function for finding head pose direction
def get_head_pose_direction(face_region, face_mesh):
    image_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    face_2d = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * 100), int(lm.y * 100)
                if idx == 1:
                    nose_x, nose_y = x, y
                face_2d.append([x, y])
            if nose_x < 40:
                direction = "Left"
            elif nose_x > 60:
                direction = "Right"
            elif nose_y < 40:
                direction = "Up"
            elif nose_y > 60:
                direction = "Down"
            else:
                direction = "Forward"
            
            return direction


# Enhance image
def enhance_image(image):
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return image


# Detect eye contact using Tensorflow CNN model
def detect_eye_contact(right_eye, left_eye, model):
    right_eye = enhance_image(right_eye)
    left_eye = enhance_image(left_eye)
    right_eye = cv2.resize(right_eye, (60, 60))
    left_eye = cv2.resize(left_eye, (60, 60))
    right_eye = right_eye / 255
    left_eye = left_eye / 255
    right_eye = tf.expand_dims(right_eye, axis=0)
    left_eye = tf.expand_dims(left_eye, axis=0)
    if right_eye.shape[-1] == 1:
        right_eye = tf.image.grayscale_to_rgb(right_eye)
    if left_eye.shape[-1] == 1:
        left_eye = tf.image.grayscale_to_rgb(left_eye)
    batch = tf.concat([right_eye, left_eye], axis=0)
    labels = ['ClosingEye', 'EyeContact', 'LeftLook', 'RightLook']
    predictions = model.predict(batch)

    right_eye_pred_label = labels[int(predictions[0].argmax(axis=-1))]
    left_eye_pred_label = labels[int(predictions[1].argmax(axis=-1))]
    
    if right_eye_pred_label == "EyeContact" or left_eye_pred_label == "EyeContact":
        return True
    else:
        return False
    

    
