#Step 1: Real Time Pose Detection (reading in what the human is doing and coverting it to something code understands)
import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Capture video (probably needs to fixed given external campera)
cap = cv2.VideoCapture("/home/trietmar/Downloads/Rasputin.mp4")  # Replace '0' with your external camera index if needed

def find_angle(x1, y1, x2, y2, x3, y3):
    # Calculate vectors
    v1 = [x2 - x1, y2 - y1]
    v2 = [x3 - x1, y3 - y1]
    
    # Calculate angle between vectors
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(i**2 for i in v1))
    magnitude2 = math.sqrt(sum(i**2 for i in v2))
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle = math.acos(cosine_angle)
    
    return math.degrees(angle)

def measure_joint_angle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    x3, y3 = landmark3.x, landmark3.y
    return find_angle(x1, y1, x2, y2, x3, y3)

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

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        hip_knee_ankle_angle = measure_joint_angle(hip, knee, ankle)

        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        shoulder_elbow_wrist_angle = measure_joint_angle(shoulder, elbow, wrist)

        cv2.putText(frame, f"Hip-Knee-Ankle Angle: {hip_knee_ankle_angle:.2f}°", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Shoulder-Elbow-Wrist Angle: {shoulder_elbow_wrist_angle:.2f}°", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        mp_drawing.draw_landmarks(
            frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display the frame
    cv2.imshow("Pose Detection", frame_bgr)



    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

[]

#Step 2: Defining Dance Moves (we only really need one for our demo; I just combined random stuff together to show how it would be to define moves though)
def is_hands_up(landmarks):
    #Check if both hands are above the shoulders.
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y

def is_squat(landmarks):
    #Check if the knees are below the hips.
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    return left_knee.y > left_hip.y and right_knee.y > right_hip.y
#Step 3: Real time Pose Recognition 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Check for "hands up"
        if is_hands_up(results.pose_landmarks):
            cv2.putText(frame_bgr, "Hands Up!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check for "squat"
        if is_squat(results.pose_landmarks):
            cv2.putText(frame_bgr, "Squat!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Recognition", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#Step 4: Synchronzing as Music
import pygame
pygame.mixer.init()
pygame.mixer.music.load("song.mp3")
pygame.mixer.music.play()
"""