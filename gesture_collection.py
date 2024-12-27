import cv2
import mediapipe as mp
import json
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture labels
gesture_labels = ["Gesture 0", "Gesture 1", "Gesture 2", "Gesture 3", "Gesture 4"]

# Data file
data_file = "gesture_data.json"
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        gesture_data = json.load(f)
else:
    gesture_data = {label: [] for label in gesture_labels}

# Start video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

gesture_idx = 0  # Start with Gesture 0
print(f"Collecting data for {gesture_labels[gesture_idx]} (Press 'n' for next)")

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
            gesture_data[gesture_labels[gesture_idx]].append(landmarks)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display current gesture
    cv2.putText(frame, f"Collecting: {gesture_labels[gesture_idx]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Next gesture
        gesture_idx = (gesture_idx + 1) % len(gesture_labels)
        print(f"Collecting data for {gesture_labels[gesture_idx]}")

cap.release()
cv2.destroyAllWindows()

# Save data
with open(data_file, "w") as f:
    json.dump(gesture_data, f)
print("Gesture data saved.")
