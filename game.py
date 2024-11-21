import subprocess
import time
import random
#import cv2
#import mediapipe as mp

f = open("angles.txt", "w")
f.close()

score = random.randint(100, 40000)

process2 = subprocess.Popen(['python', 'DefaultDanceTracking.py'])

print(f'Your score is {score}!')

"""
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("C:\\Users\\aadwu\\Downloads\\YouWon.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        cv2.putText(frame_bgr, f"You scored {points} points!", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        mp_drawing.draw_landmarks(
            frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Points Screen", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

#process1.wait()
#process2.wait()
#process3.wait()