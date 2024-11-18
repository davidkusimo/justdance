import subprocess
import time
import cv2
import mediapipe as mp

points = 0

process1 = subprocess.Popen(['python', 'Camera.py'])
process2 = subprocess.Popen(['python', 'RasputinTracking.py'])

while(True):
    f = open("angles.txt", "r")

    angles1 = f.read().split(',')
    angles2 = f.read().split(',')

    difference = [abs(float(angles1[0]) - float(angles2[0])), abs(float(angles1[1]) - float(angles2[1]))]

    if difference[0] <= 45 and difference[1] <= 45:
        points += 100

    f.close()
    f.open("angles.txt", "w")
    f.close

    time.sleep(4)

    if(process2.poll() is None):
        process1.terminate()
        break

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("C:\\Users\\aadwu\\Downloads\\YouWon.mp3")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        cv2.putText(frame_bgr, f"You scored {points} points!", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        mp_drawing.draw_landmarks(
            frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Detection", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#process1.wait()
#process2.wait()