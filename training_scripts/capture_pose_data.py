#This uses MediaPipe Pose to track 33 landmarks.
#Each frame’s landmarks are stored in a CSV along with the pose label.
#You’ll run this for each pose and store CSVs in the right folder (regular, blind_accessible, disabled_accessible).




#old version to train only from camera
import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

data = []
pose_name = "tree_pose"  # Change this per recording session

cap = cv2.VideoCapture(0)  # Webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks.append(pose_name)
        data.append(landmarks)
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
columns = [f"{coord}_{i}" for i in range(33) for coord in ("x", "y", "z")] + ["label"]
df = pd.DataFrame(data, columns=columns)
os.makedirs("dataset/regular", exist_ok=True)
df.to_csv(f"dataset/regular/{pose_name}.csv", index=False) 



#updated version can capture from camera and video

# import cv2
# import mediapipe as mp
# import csv
# import os
# import argparse

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def capture_pose_data(source, pose_name, use_camera=True):
#     # Create folder for pose
#     save_dir = os.path.join("data", pose_name)
#     os.makedirs(save_dir, exist_ok=True)
#     csv_path = os.path.join(save_dir, f"{pose_name}_data.csv")

#     cap = cv2.VideoCapture(source)

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         with open(csv_path, mode='w', newline='') as file:
#             csv_writer = csv.writer(file)

#             # Write header row
#             landmarks = [f"x{i}" for i in range(33)] + [f"y{i}" for i in range(33)] + [f"z{i}" for i in range(33)]
#             csv_writer.writerow(landmarks)

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False
#                 results = pose.process(image)

#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 if results.pose_landmarks:
#                     # Draw landmarks on screen
#                     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#                     # Extract keypoints
#                     keypoints = []
#                     for lm in results.pose_landmarks.landmark:
#                         keypoints.append(lm.x)
#                         keypoints.append(lm.y)
#                         keypoints.append(lm.z)

#                     csv_writer.writerow(keypoints)

#                 cv2.imshow('Pose Capture', image)

#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"✅ Data saved to {csv_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Capture Yoga Pose Data")
#     parser.add_argument("--pose_name", type=str, required=True, help="Name of the yoga pose")
#     parser.add_argument("--camera", type=int, help="Camera index (default=0)")
#     parser.add_argument("--video", type=str, help="Path to video file")

#     args = parser.parse_args()

#     if args.camera is not None:
#         capture_pose_data(args.camera, args.pose_name, use_camera=True)
#     elif args.video is not None:
#         capture_pose_data(args.video, args.pose_name, use_camera=False)
#     else:
#         print("❌ Please specify either --camera or --video")



#to run from camera 0 → default webcam ,If using DroidCam/Iriun, select their camera index (try 1, 2, etc.)
#command - python training_scripts/capture_pose_data.py --pose_name Warrior --camera 0

#to capture from video
#command - python training_scripts/capture_pose_data.py --pose_name Warrior --video my_pose_video.mp4


