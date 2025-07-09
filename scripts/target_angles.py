import cv2
import os
import joblib
import numpy as np
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.canonical_poses import asymmetrical_poses, all_poses, pose_to_rule, flip_pose
from scripts.utils import generate_keypoints, extract_joint_angles

# After flipping the keypoints if necessary, extract the keypoint values and average to form the target
# Identify key joints for each pose

pose_key_joints = {
    "Boat_Pose_or_Paripurna_Navasana_": ["left_knee", "right_knee", "left_hip", "right_hip", "torso_bend"],
    "Bound_Angle_Pose_or_Baddha_Konasana_": ["left_knee", "right_knee", "left_hip", "right_hip"],
    "Bow_Pose_or_Dhanurasana_": ["left_knee", "right_knee", "left_shoulder", "right_shoulder"],
    "Bridge_Pose_or_Setu_Bandha_Sarvangasana_": ["left_knee", "right_knee", "hip_tilt", "torso_bend"],
    "Camel_Pose_or_Ustrasana_": ["left_hip", "right_hip", "left_shoulder", "right_shoulder", "torso_bend"],
    "Cat_Cow_Pose_or_Marjaryasana_": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "torso_bend"],
    "Chair_Pose_or_Utkatasana_": ["left_knee", "right_knee", "left_hip", "right_hip", "shoulder_tilt"],
    "Child_Pose_or_Balasana_": ["left_knee", "right_knee", "torso_bend"],
    "Cobra_Pose_or_Bhujangasana_": ["left_elbow", "right_elbow", "torso_bend", "shoulder_tilt"],
    "Cockerel_Pose": ["left_knee", "right_knee", "shoulder_tilt"],
    "Corpse_Pose_or_Savasana_": [],
    "Cow_Face_Pose_or_Gomukhasana_": ["left_knee", "right_knee", "left_elbow", "right_elbow"],
    "Crane_(Crow)_Pose_or_Bakasana_": ["left_elbow", "right_elbow", "torso_bend"],
    "Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Dolphin_Pose_or_Ardha_Pincha_Mayurasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_": ["left_knee", "right_knee", "shoulder_tilt", "hip_tilt"],
    "Eagle_Pose_or_Garudasana_": ["left_knee", "right_knee", "left_elbow", "right_elbow"],
    "Eight-Angle_Pose_or_Astavakrasana_": ["left_elbow", "right_elbow", "torso_bend"],
    "Extended_Puppy_Pose_or_Uttana_Shishosana_": ["left_shoulder", "right_shoulder", "torso_bend"],
    "Feathered_Peacock_Pose_or_Pincha_Mayurasana_": ["left_elbow", "right_elbow", "shoulder_tilt"],
    "Firefly_Pose_or_Tittibhasana_": ["left_knee", "right_knee", "left_elbow", "right_elbow"],
    "Fish_Pose_or_Matsyasana_": ["left_hip", "right_hip", "torso_bend"],
    "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_": ["left_elbow", "right_elbow", "shoulder_tilt"],
    "Frog_Pose_or_Bhekasana": ["left_knee", "right_knee", "hip_tilt"],
    "Garland_Pose_or_Malasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Handstand_pose_or_Adho_Mukha_Vrksasana_": ["left_shoulder", "right_shoulder", "shoulder_tilt"],
    "Happy_Baby_Pose_or_Ananda_Balasana_": ["left_knee", "right_knee", "left_hip", "right_hip"],
    "Legs-Up-the-Wall_Pose_or_Viparita_Karani_": ["left_hip", "right_hip", "hip_tilt"],
    "Locust_Pose_or_Salabhasana_": ["left_knee", "right_knee", "torso_bend"],
    "Peacock_Pose_or_Mayurasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Pigeon_Pose_or_Kapotasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Plank_Pose_or_Kumbhakasana_": ["left_shoulder", "right_shoulder", "hip_tilt"],
    "Plow_Pose_or_Halasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Scale_Pose_or_Tolasana_": ["left_hip", "right_hip", "torso_bend"],
    "Scorpion_pose_or_vrischikasana": ["left_knee", "right_knee", "shoulder_tilt"],
    "Seated_Forward_Bend_pose_or_Paschimottanasana_": ["left_knee", "right_knee", "torso_bend"],
    "Shoulder-Pressing_Pose_or_Bhujapidasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Sitting pose 1 (normal)": ["left_knee", "right_knee", "hip_tilt"],
    "Split pose": ["left_knee", "right_knee", "hip_tilt"],
    "Staff_Pose_or_Dandasana_": ["left_knee", "right_knee", "torso_bend"],
    "Standing_Forward_Bend_pose_or_Uttanasana_": ["left_knee", "right_knee", "torso_bend"],
    "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Supported_Headstand_pose_or_Salamba_Sirsasana_": ["shoulder_tilt", "hip_tilt"],
    "Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_": ["left_hip", "right_hip", "shoulder_tilt"],
    "Supta_Baddha_Konasana_": ["left_knee", "right_knee"],
    "Supta_Virasana_Vajrasana": ["left_knee", "right_knee", "hip_tilt"],
    "Tortoise_Pose": ["left_knee", "right_knee", "torso_bend"],
    "Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_": ["left_elbow", "right_elbow", "left_knee", "right_knee"],
    "Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Upward_Plank_Pose_or_Purvottanasana_": ["left_elbow", "right_elbow", "hip_tilt"],
    "Virasana_or_Vajrasana": ["left_knee", "right_knee", "hip_tilt"],
    "Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_": ["left_knee", "right_knee", "torso_bend"],
    "Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_": ["left_knee", "right_knee", "torso_bend"],
    "Wind_Relieving_pose_or_Pawanmuktasana": ["left_knee", "right_knee"],
    "Yogic_sleep_pose": ["torso_bend"],

    "Akarna_Dhanurasana": ["left_knee", "left_hip", "right_knee", "shoulder_tilt"],
    "Bharadvajas_Twist_pose_or_Bharadvajasana_I_": ["left_hip", "left_shoulder", "torso_bend"],
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_": ["left_knee", "left_hip", "right_knee", "right_shoulder"],
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_": ["left_knee", "left_hip", "right_knee", "left_shoulder", "right_shoulder"],
    "Gate_Pose_or_Parighasana_": ["left_knee", "left_hip", "right_knee"],
    "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_": ["left_hip", "right_hip", "shoulder_tilt"],
    "Half_Moon_Pose_or_Ardha_Chandrasana_": ["left_knee", "left_hip", "right_ankle", "hip_tilt"],
    "Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_": ["left_knee", "right_knee", "torso_bend"],
    "Heron_Pose_or_Krounchasana_": ["left_knee", "right_knee"],
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_": ["left_knee", "right_knee", "torso_bend"],
    "Lord_of_the_Dance_Pose_or_Natarajasana_": ["left_knee", "right_knee", "left_hip", "torso_bend"],
    "Low_Lunge_pose_or_Anjaneyasana_": ["left_knee", "left_hip", "right_knee"],
    "Noose_Pose_or_Pasasana_": ["left_knee", "torso_bend", "left_shoulder"],
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II": ["left_knee", "right_knee", "left_elbow", "right_elbow", "torso_bend"],
    "Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_": ["left_knee", "right_knee", "shoulder_tilt"],
    "Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_": ["left_elbow", "right_elbow", "torso_bend"],
    "Side_Plank_Pose_or_Vasisthasana_": ["left_elbow", "left_shoulder", "right_hip"],
    "Side-Reclining_Leg_Lift_pose_or_Anantasana_": ["left_knee", "right_knee", "hip_tilt"],
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana": ["left_knee", "right_knee", "left_hip", "hip_tilt"],
    "Tree_Pose_or_Vrksasana_": ["left_knee", "left_hip", "right_knee"],
    "viparita_virabhadrasana_or_reverse_warrior_pose": ["left_knee", "right_knee", "torso_bend"],
    "Warrior_I_Pose_or_Virabhadrasana_I_": ["left_knee", "left_hip", "right_knee", "shoulder_tilt"],
    "Warrior_II_Pose_or_Virabhadrasana_II_": ["left_knee", "left_hip", "right_knee", "torso_bend"],
    "Warrior_III_Pose_or_Virabhadrasana_III_": ["left_knee", "right_hip", "right_knee", "hip_tilt"],
    "Wild_Thing_pose_or_Camatkarasana_": ["left_knee", "right_knee", "torso_bend"],
    "Rajakapotasana": ["left_knee", "right_knee", "hip_tilt"]
}

model_path = 'models/pose_landmarker_heavy.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
pose_target_angles = {}

for pose in all_poses:
    joint_angle_lists = defaultdict(list)
    base_dir = 'data/raw/'
    pose_dir = os.path.join(base_dir, pose)
    for filename in os.listdir(pose_dir):
        img_path = os.path.join(pose_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        keypoints_arr_norm = generate_keypoints(image, pose_landmarker)
        if keypoints_arr_norm is None:
            continue
        if pose in asymmetrical_poses and pose_to_rule[pose](keypoints_arr_norm) == False:
            keypoints_arr_norm = flip_pose(keypoints_arr_norm)
        angles = extract_joint_angles(keypoints_arr_norm)
        if not angles:
            continue
        for joint, angle in angles.items():
            if joint in pose_key_joints[pose]:
                joint_angle_lists[joint].append(angle)
    target_angles = {joint: float(np.mean(angles)) for joint, angles in joint_angle_lists.items()}
    pose_target_angles[pose] = target_angles

joblib.dump(pose_target_angles, 'models/pose_target_angles.joblib')
