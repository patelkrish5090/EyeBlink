import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import time

# Load face and eye detection models from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("facial-landmarks-recognition-master\\facial-landmarks-recognition-master\\shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Minimum threshold for eye aspect ratio to be considered a blink
EAR_THRESHOLD = 0.2

# Initialize variables
frame_count = 0
blink_count = 0
blinking = False

l_start, l_end = 42, 48  # Indices for left eye landmarks
r_start, r_end = 36, 42  # Indices for right eye landmarks

# Start the video stream using the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = shape_to_np(shape)  # Convert dlib shape to NumPy array

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if not blinking:
                blinking = True
                blink_count += 1
                start_time = time.time()  # Record the start time of blink
        else:
            blinking = False
            start_time = None

    # ...

    if start_time:
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f"Blink {blink_count}: Elapsed time since blink: {elapsed_time:.2f} seconds")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
