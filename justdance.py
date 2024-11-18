'''
#Step 1: Real Time Pose Detection (reading in what the human is doing and coverting it to something code understands)
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Capture video (probably needs to fixed given external campera)
cap = cv2.VideoCapture(0)  # Replace '0' with your external camera index if needed

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

[]

#Step 2: Defining Dance Moves (we only really need one for our demo; I just combined random stuff together to show how it would be to define moves though)
def is_hands_up(landmarks):
    """Check if both hands are above the shoulders."""
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y

def is_squat(landmarks):
    """Check if the knees are below the hips."""
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

#Sample Script (very unfinished)
import cv2
import mediapipe as mp
import pygame
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame for music
pygame.mixer.init()
pygame.mixer.music.load("song.mp3")  # Replace with the path to your music file

# Define scoring and timing
score = 0
start_time = None

# Define move detection functions
def is_hands_up(landmarks):
    """Check if both hands are above the shoulders."""
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    return left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y

def is_squat(landmarks):
    """Check if the knees are below the hips."""
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    return left_knee.y > left_hip.y and right_knee.y > right_hip.y

# Start capturing video
cap = cv2.VideoCapture(0)  # Replace with your camera index if needed
pygame.mixer.music.play()  # Start music playback
start_time = time.time()  # Start timer

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose detection
        results = pose.process(frame_rgb)

        # Convert frame back to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Check for specific moves and update the score
            if is_hands_up(results.pose_landmarks):
                score += 10  # Add points for hands-up
                cv2.putText(frame_bgr, "Hands Up!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if is_squat(results.pose_landmarks):
                score += 10  # Add points for squat
                cv2.putText(frame_bgr, "Squat!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the pose landmarks on the frame
            mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the score
        cv2.putText(frame_bgr, f"Score: {score}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the video feed
        cv2.imshow("Dance Game", frame_bgr)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Game interrupted!")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

    print(f"Final Score: {score}")

'''