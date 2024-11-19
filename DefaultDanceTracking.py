#Step 1: Real Time Pose Detection (reading in what the human is doing and coverting it to something code understands)
import cv2
import mediapipe as mp
import math
import subprocess

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Capture video (probably needs to fixed given external campera)
cap = cv2.VideoCapture("C:\\Users\\aadwu\\Downloads\\default.mp4")  # Replace '0' with your external camera index if needed

def find_angle(x1, y1, x2, y2, x3, y3):
    #Calculate angle between vectors formed by three points.
    v1 = [x2 - x1, y2 - y1]
    v2 = [x3 - x1, y3 - y1]
    
    magnitude1 = math.sqrt(sum(i**2 for i in v1))
    magnitude2 = math.sqrt(sum(i**2 for i in v2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(max(-1, min(1, cosine_angle)))
    return math.degrees(angle)

def measure_joint_angle(landmark1, landmark2, landmark3):
    #Calculate angle between three pose landmarks.
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y
    return find_angle(x1, y1, x2, y2, x3, y3)

counter = 0

process1 = subprocess.Popen(['python', 'Camera.py'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect poses
    results = pose.process(frame_rgb)

    # Convert back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) 


    right_arm_angle = 0
    left_arm_angle = 0

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Left arm (shoulder-elbow-wrist)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        left_arm_angle = measure_joint_angle(left_shoulder, left_elbow, left_wrist)

        # Right arm (shoulder-elbow-wrist)
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_arm_angle = measure_joint_angle(right_shoulder, right_elbow, right_wrist)

        # Display angles on screen
        cv2.putText(frame_bgr, f"Left Arm Angle: {left_arm_angle:.2f}°", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Right Arm Angle: {right_arm_angle:.2f}°", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        mp_drawing.draw_landmarks(
            frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    if counter % 100 == 0:
        f = open("angles.txt", "a") #appends
        f.write(f'{right_arm_angle},{left_arm_angle}\n')
        f.close()

    # Display the frame
    cv2.imshow("Pose Detection", frame_bgr)
    counter += 1

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

process1.kill()

f = open("angles.txt", "r")

list = []
score = 0

for line in f:
    list += [line.split()]

for i in range(1, len(list)):
    if abs(int(list[i][0]) - int(list[i-1][0])) <= 45 and abs(int(list[i][1]) - int(list[i-1][1])) <= 45:
        score += 100

print(f"Your score is {score}!")

cap.release()
cv2.destroyAllWindows()
