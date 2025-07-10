# Extract and normalise keypoints. Split into train/val/test

import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from scripts.utils import generate_keypoints, normalise_keypoints

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

        keypoints_arr = generate_keypoints(image, pose_landmarker)
        keypoints_arr_norm = normalise_keypoints(keypoints_arr)
        if keypoints_arr_norm is None:
            continue
        pose_name = os.path.basename(dirpath)
        row = [pose_name] + keypoints_arr_norm.flatten().tolist()  # (pose_name, x1, y1, z1, v1, x2, y2, z2,...)
        rows.append(row)

df = pd.DataFrame(rows, columns=colnames).drop_duplicates()
df.to_parquet('data/processed/df.parquet')

x = df.drop('label', axis=1)
y = df['label']
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25)
x_train = x_train.reset_index(drop=True)
x_val = x_val.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

x_train.to_parquet('data/processed/x_train.parquet')
x_val.to_parquet('data/processed/x_val.parquet')
x_test.to_parquet('data/processed/x_test.parquet')
pd.DataFrame(y_train, columns=['label']).to_parquet('data/processed/y_train.parquet')
pd.DataFrame(y_val, columns=['label']).to_parquet('data/processed/y_val.parquet')
pd.DataFrame(y_test, columns=['label']).to_parquet('data/processed/y_test.parquet')