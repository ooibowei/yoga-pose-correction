# Extract and normalise keypoints. Split into train/val/test

import cv2
import os
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Use heavy model to train pose classifier
model_path = 'models/pose_landmarker_heavy.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

data_dir = 'data/raw'
rows = []
colnames = ['label'] + [f'{dim}{i}' for i in range(33) for dim in ('x','y','z','v')]
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in filenames:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(dirpath, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        pose_res = pose_landmarker.detect(mp_image)
        if not pose_res.pose_landmarks:
            continue

        keypoints = []  # (x1, y1, z1, v1, x2, y2, z2,...)
        for lm in pose_res.pose_landmarks[0]:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        pose_name = os.path.basename(dirpath)
        row = [pose_name] + keypoints
        rows.append(row)

df = pd.DataFrame(rows, columns=colnames).drop_duplicates()
df.to_parquet('data/processed/df.parquet')

"""
Normalize keypoints wrt reference landmark (eg hip)
Decide what to do with z-coord
Train/val/test split
"""