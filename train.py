import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load data
data_file = "gesture_data.json"
with open(data_file, "r") as f:
    gesture_data = json.load(f)

X, y = [], []
for label_idx, label in enumerate(gesture_data.keys()):
    for sample in gesture_data[label]:
        X.append(np.array(sample).flatten())
        y.append(label_idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(gesture_data.keys()))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(gesture_data.keys()), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Save model
model.save("gesture_recognition_model.h5")
print("Model training complete and saved.")
