# Real-Time Yoga Pose Correction App

This is a real-time yoga pose correction app that detects key body joints and provides feedback for pose correction. The app supports both webcam streaming and file uploads (image/video)

The app features a pipeline that extracts a user's keypoints using MediaPipe and predicts the target pose using a feedforward neural network. These keypoints are then compared against reference angles of key joints for the pose, generating visual and spoken corrections for the user

The app includes a FastAPI backend using REST APIs, a lightweight HTML/JS frontend and asynchronous audio feedback using Edge TTS. It uses GitLab CI for continuous integration and is designed for local deployment 

## Key Features

- Pose estimation with MediaPipe
- Feedforward Neural Network with PyTorch
- 

## Build 3 models
1. Pose Estimation: First, use a pose estimation model to track the keypoints of the user's body in real-time.
2. Pose Classification: Then, use a pose classification model to identify which yoga pose the user is trying to perform.
3. Pose Correction: Finally, apply a pose correction model to analyze any deviations from the correct pose and offer feedback to help the user adjust their alignment.

## Pose Estimation Model
What: A pre-built model that can find and track keypoints (like shoulders and hips) in an image or video.
MediaPipe: A popular tool for detecting body keypoints.
Why: This model helps extract the keypoints from the user's video feed, which is essential for understanding their pose.

## Pose Classification Model
What: A model that sorts different poses into categories based on the keypoints
Why: To determine which yoga pose the user is performing.
Use MLP with keypoints
Augment keypoints training set 
- normalise keypoints
- Add small random noise (jitter)
- Random rotation (around center)
- Random scaling
- Random translation
- Random dropout (simulate missing landmarks)
No need to augment images

## Post Correction Model
What: A system that provides suggestions on how to adjust the user's pose to make it correct.
Rule-Based (start with this): Uses predefined angle thresholds and logic to detect pose errors directly from joint keypoints
ML Model (future): Trains a model to recognize and correct pose errors by learning patterns from synthetically altered keypoints of correct pose (perturbations)
Why: To give users specific advice on how to fix their pose based on detected mistakes.

## Pose Dataset: 
https://www.kaggle.com/datasets/akashrayhan/yoga-82/

Deployment?