import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model and labels
model = load_model("gesture_recognition_model.h5")
gesture_labels = ["Gesture 0", "Gesture 1", "Gesture 2", "Gesture 3", "Gesture 4"]

# Start video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            feature_vector = np.array(landmarks).flatten().reshape(1, -1)

            # Predict gesture
            prediction = model.predict(feature_vector)
            predicted_idx = np.argmax(prediction)
            predicted_label = gesture_labels[predicted_idx]

            # Draw landmarks and predicted label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
