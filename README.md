Hand Gesture Recognition Project
This project implements a hand gesture recognition system using Python, MediaPipe, and TensorFlow/Keras. The system captures hand landmarks, trains a machine learning model to classify gestures, and performs real-time gesture recognition. It is designed to be modular and extendable for various applications like sign language recognition, touchless interfaces, or gesture-based controls.

Project Structure
Data Collection

Uses MediaPipe to detect hand landmarks.
Captures and saves labeled gesture data for training.
Supports multiple gestures with real-time visualization.
Saves data as a JSON file.
Model Training

Trains a neural network using TensorFlow/Keras.
Uses collected data to create a gesture recognition model.
Achieves high accuracy with configurable layers and epochs.
Real-Time Gesture Recognition

Recognizes gestures in real time using the trained model.
Displays the predicted gesture with visual feedback on live video feed.
Key Features
Real-Time Processing: MediaPipe ensures low-latency hand tracking.
Customizable Gestures: Easily add or modify gesture labels.
Modular Design: Separate scripts for data collection, model training, and recognition.
High Accuracy: Neural network achieves precise classification with sufficient training data.
Interactive Visuals: Real-time display of landmarks and predictions.
How to Run
Data Collection:

Run gesture_collection.py to collect labeled gesture data.
Press 'n' to switch between gestures during collection.
Collect at least 100 samples per gesture for optimal accuracy.
Model Training:

Run model_training.py to train the neural network.
Adjust hyperparameters as needed (e.g., epochs, batch size).
Gesture Recognition:

Run gesture_recognition.py for real-time gesture prediction.
The script uses your webcam for live input.
