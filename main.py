import cv2
import pandas as pd
import time
from datetime import datetime
from utils import *

# If you want to use Yolov8 model to detect head position
# import Yolov8Model as yolov8

# Load the models
detector, model, face_mesh = load_models()
# Tracking eye contact time
user_eye_contact_data = []
# Initialize a user count
user_count = 0

def detectFacEye(image, conf, eye_crop_size=10):
    global user_count, start_time
    height, width, _ = image.shape
    detector.setInputSize((width, height))
    
    _, faces = detector.detect(image)
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            
            right_eye_x, right_eye_y = int(face[4]), int(face[5])
            left_eye_x, left_eye_y = int(face[6]), int(face[7])
            right_eye = image[right_eye_y - eye_crop_size:right_eye_y + eye_crop_size, right_eye_x - eye_crop_size:right_eye_x + eye_crop_size]
            left_eye = image[left_eye_y - eye_crop_size:left_eye_y + eye_crop_size, left_eye_x - eye_crop_size:left_eye_x + eye_crop_size]
            eye_contact = detect_eye_contact(right_eye, left_eye, model)
            
            factor = 0.9
            expand_x = int(factor * w)
            expand_y = int(factor * h)
            new_x = max(0, x - expand_x)
            new_y = max(0, y - expand_y)
            new_w = min(width, w + 2 * expand_x)
            new_h = min(height, h + 2 * expand_y)

            face_region = image[new_y:new_y + new_h, new_x:new_x + new_w]
            face_region = enhance_image(face_region)
            # pred = detect_face_direction(face_region) # in yolov8
            pred = get_head_pose_direction(face_region, face_mesh)
            
            # if pred in ['Front', 'FrontLeft', 'FrontRight']: # in yolov8
            if pred in ["Forward", "Up"]:
                if eye_contact:
                    # Log the eye contact time for this user
                    elapsed_time = time.time() - start_time
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    user_count += 1
                    user_id = f"user_{user_count}"
                    
                    user_eye_contact_data.append({
                        "User ID": user_id,
                        "Eye Contact Duration (seconds)": elapsed_time,
                        "Date and Time": current_time
                    })
                    
                    color = (0, 255, 0)
                    text = f"EyeContact {user_id} {elapsed_time:.2f}s {pred}"
                    
                    start_time = time.time()
                else:
                        color = (0, 0, 255)
                        text = f"NoEyeContact {pred}"
            else:
                color = (0, 0, 255)
                text = f"NoEyeContact {pred}"
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (right_eye_x - 5, right_eye_y - 5), (right_eye_x + 5, right_eye_y + 5), color, 2)
            cv2.rectangle(image, (left_eye_x - 5, left_eye_y - 5), (left_eye_x + 5, left_eye_y + 5), color, 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
def process_video(input_video_path, conf, eye_crop_size=10):
    global start_time
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    # Initialize start time
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detectFacEye(frame, conf=conf, eye_crop_size=eye_crop_size)
        
        cv2.imshow('Face-Detection', frame)
        print('Processed frame {}/{}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES), total_frames))
        
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save the eye contact data to an Excel file
    df = pd.DataFrame(user_eye_contact_data)
    df.to_excel("data.xlsx", index=False)
    print("Eye contact times saved to eye_contact_times.xlsx")

if __name__ == "__main__":
    input_video_path = 0
    output_video_path = 'output.avi'
    
    process_video(input_video_path, conf=0.5, eye_crop_size=10)