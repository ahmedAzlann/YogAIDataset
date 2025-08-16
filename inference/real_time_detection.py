#Loads your trained model & label encoder.
#Uses webcam to detect poses in real time.
#Predicts pose name & gives a correction message if confidence is low.

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pickle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load model & label encoder
model = tf.keras.models.load_model("models/regular_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        prediction = model.predict(np.array([landmarks]))
        class_id = np.argmax(prediction)
        pose_name = le.inverse_transform([class_id])[0]
        confidence = prediction[0][class_id]

        cv2.putText(frame, f"{pose_name} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if confidence < 0.7:
            cv2.putText(frame, "Adjust your form!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Yoga Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
