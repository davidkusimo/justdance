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
    x1, y1 = landmark1[0], landmark1[1]
    x2, y2 = landmark2[0], landmark2[1]
    x3, y3 = landmark3[0], landmark3[1]
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

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        hip_knee_ankle_angle = measure_joint_angle(hip, knee, ankle)

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

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
