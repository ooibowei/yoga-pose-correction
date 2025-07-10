import cv2
import os
import joblib
import numpy as np
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.utils import generate_keypoints, normalise_keypoints, extract_joint_angles, all_poses, pose_to_rule, flip_pose, pose_key_joints

model_path = 'models/pose_landmarker_heavy.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
pose_target_angles = {}

# Flip the keypoints if necessary to the canonical side then and average them to form the target
for pose in all_poses:
    joint_angle_lists = defaultdict(list)
    base_dir = 'data/raw/'
    pose_dir = os.path.join(base_dir, pose)
    for filename in os.listdir(pose_dir):
        img_path = os.path.join(pose_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        keypoints_arr = generate_keypoints(image, pose_landmarker)
        keypoints_arr_norm = normalise_keypoints(keypoints_arr)
        if keypoints_arr_norm is None:
            continue
        if pose in pose_to_rule and pose_to_rule[pose](keypoints_arr_norm) == False:
            keypoints_arr_norm = flip_pose(keypoints_arr_norm)
        angles = extract_joint_angles(keypoints_arr_norm)
        if not angles:
            continue
        for joint, angle in angles.items():
            joint_angle_lists[joint].append(angle)
    target_angles = {joint: float(np.mean(angles)) for joint, angles in joint_angle_lists.items()}
    pose_target_angles[pose] = target_angles
joblib.dump(pose_target_angles, 'models/pose_target_angles.joblib')

# Extract the keypoint values of key joints
pose_target_key_angles = {}
for pose in pose_target_angles:
    target_key_angles = {joint: angle for joint, angle in pose_target_angles[pose].items() if joint in pose_key_joints[pose]}
    pose_target_key_angles[pose] = target_key_angles
joblib.dump(pose_target_key_angles, 'models/pose_target_key_angles.joblib')