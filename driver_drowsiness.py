import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("C:\\Users\\ManU\\Downloads\\DrowsinessDetector\\shape_predictor_68_face_landmarks.dat")
    print("Landmark predictor loaded successfully.")
except Exception as e:
    print(f"Error loading landmark predictor: {e}")
    exit()

# Status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    print(f"Eye Aspect Ratio: {ratio}")  # Debug statement
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(f"Number of faces detected: {len(faces)}")  # Debug statement

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        print(f"Landmarks: {landmarks}")  # Debug statement

        # Detect blinking for both eyes
        left_blink = blinked(landmarks[36], landmarks[37], 
                          landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], 
                           landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Determine the current state (sleep, drowsy, active)
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display the status on the frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks on the face
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()