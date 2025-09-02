import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
from car_simulation import drowsy_detected


# 🎵 Initialize Pygame mixer and load beep sound
pygame.mixer.init()
sound = pygame.mixer.Sound(r"C:\Users\ManU\Downloads\DrowsinessDetector\data\beep.wav")

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
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

# 🔔 Beeper control
beep_running = False

def start_beep():
    global beep_running
    if not beep_running:
        beep_running = True
        sound.play(loops=-1)  # continuous loop

def stop_beep():
    global beep_running
    if beep_running:
        beep_running = False
        sound.stop()

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25: return 2      # open
    elif ratio > 0.21: return 1    # semi-open / drowsy
    else: return 0                 # closed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], 
                          landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], 
                           landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                start_beep()   # 🔔 start continuous beep

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                start_beep()   # 🔔 start continuous beep

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                stop_beep()   # 🔕 stop beep when active

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

stop_beep()  # 🔕 ensure beep stops before exit
cap.release()
cv2.destroyAllWindows()
