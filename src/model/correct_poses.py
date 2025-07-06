import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.utils import generate_keypoints, extract_joint_angles

# For each pose, need to decide if it should be flipped to the 'canonical' side
# After flipping the keypoints, then extract the keypoint values and average to form the target
# For inference, use an image to predict. Then check if it should be flipped to the 'canonical' side
# After flipping, compare the angles and give corrections (flipping the side if the image was flipped)

model_path = 'models/pose_landmarker_heavy.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
image = cv2.imread('data/raw/Warrior_II_Pose_or_Virabhadrasana_II_/Warrior_II_Pose_or_Virabhadrasana_II__image_8.jpg')
keypoints_arr_norm = generate_keypoints(image, pose_landmarker)
angles = extract_joint_angles(keypoints_arr_norm)

import os
import numpy as np
from collections import defaultdict
joint_angle_lists = defaultdict(list)
pose_dir = 'data/raw/Warrior_II_Pose_or_Virabhadrasana_II_/'
for filename in os.listdir(pose_dir):
    img_path = os.path.join(pose_dir, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue
    keypoints_arr_norm = generate_keypoints(image, pose_landmarker)
    if keypoints_arr_norm is None:
        continue
    angles = extract_joint_angles(keypoints_arr_norm)
    if not angles:
        continue
    for joint, angle in angles.items():
        joint_angle_lists[joint].append(angle)
ideal_angles = {
    joint: float(np.mean(angles)) for joint, angles in joint_angle_lists.items() if len(angles) >= 5
}
print(ideal_angles)


# Symmetrical poses that don't need mirroring
symmetrical_poses = {
    "Boat_Pose_or_Paripurna_Navasana_",
    "Bound_Angle_Pose_or_Baddha_Konasana_",
    "Bow_Pose_or_Dhanurasana_",
    "Bridge_Pose_or_Setu_Bandha_Sarvangasana_",
    "Camel_Pose_or_Ustrasana_",
    "Cat_Cow_Pose_or_Marjaryasana_",
    "Chair_Pose_or_Utkatasana_",
    "Child_Pose_or_Balasana_",
    "Cobra_Pose_or_Bhujangasana_",
    "Cockerel_Pose",
    "Corpse_Pose_or_Savasana_",
    "Cow_Face_Pose_or_Gomukhasana_",
    "Crane_(Crow)_Pose_or_Bakasana_",
    "Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_",
    "Dolphin_Pose_or_Ardha_Pincha_Mayurasana_",
    "Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_",
    "Eagle_Pose_or_Garudasana_",
    "Eight-Angle_Pose_or_Astavakrasana_",
    "Extended_Puppy_Pose_or_Uttana_Shishosana_",
    "Feathered_Peacock_Pose_or_Pincha_Mayurasana_",
    "Firefly_Pose_or_Tittibhasana_",
    "Fish_Pose_or_Matsyasana_",
    "Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana_",
    "Frog_Pose_or_Bhekasana",
    "Garland_Pose_or_Malasana_",
    "Handstand_pose_or_Adho_Mukha_Vrksasana_",
    "Happy_Baby_Pose_or_Ananda_Balasana_",
    "Heron_Pose_or_Krounchasana_",
    "Legs-Up-the-Wall_Pose_or_Viparita_Karani_",
    "Locust_Pose_or_Salabhasana_",
    "Peacock_Pose_or_Mayurasana_",
    "Plank_Pose_or_Kumbhakasana_",
    "Plow_Pose_or_Halasana_",
    "Scale_Pose_or_Tolasana_",
    "Scorpion_pose_or_vrischikasana",
    "Seated_Forward_Bend_pose_or_Paschimottanasana_",
    "Shoulder-Pressing_Pose_or_Bhujapidasana_",
    "Sitting pose 1 (normal)",
    "Split pose",
    "Staff_Pose_or_Dandasana_",
    "Standing_Forward_Bend_pose_or_Uttanasana_",
    "Standing_Split_pose_or_Urdhva_Prasarita_Eka_Padasana_",
    "Supported_Headstand_pose_or_Salamba_Sirsasana_",
    "Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_",
    "Supta_Baddha_Konasana_",
    "Supta_Virasana_Vajrasana",
    "Tortoise_Pose",
    "Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana_",
    "Upward_Facing_Two-Foot_Staff_Pose_or_Dwi_Pada_Viparita_Dandasana_",
    "Upward_Plank_Pose_or_Purvottanasana_",
    "Virasana_or_Vajrasana",
    "Wide-Angle_Seated_Forward_Bend_pose_or_Upavistha_Konasana_",
    "Wide-Legged_Forward_Bend_pose_or_Prasarita_Padottanasana_",
    "Wind_Relieving_pose_or_Pawanmuktasana",
    "Yogic_sleep_pose"
}

asymmetrical_poses = {
    "Akarna_Dhanurasana",
    "Bharadvajas_Twist_pose_or_Bharadvajasana_I_",
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_",
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_",
    "Gate_Pose_or_Parighasana_",
    "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_",
    "Half_Moon_Pose_or_Ardha_Chandrasana_",
    "Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_",
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_",
    "Lord_of_the_Dance_Pose_or_Natarajasana_",
    "Low_Lunge_pose_or_Anjaneyasana_",
    "Noose_Pose_or_Pasasana_",
    "Pigeon_Pose_or_Kapotasana_",
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II",
    "Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_",
    "Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_",
    "Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_",
    "Side_Plank_Pose_or_Vasisthasana_",
    "Side-Reclining_Leg_Lift_pose_or_Anantasana_",
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana",
    "Tree_Pose_or_Vrksasana_",
    "viparita_virabhadrasana_or_reverse_warrior_pose",
    "Warrior_I_Pose_or_Virabhadrasana_I_",
    "Warrior_II_Pose_or_Virabhadrasana_II_",
    "Warrior_III_Pose_or_Virabhadrasana_III_",
    "Wild_Thing_pose_or_Camatkarasana_",
    "Rajakapotasana"
}

def is_left_hip_forward(keypoints):
    return keypoints[23].x < keypoints[24].x

def is_left_wrist_more_left(keypoints):
    return keypoints[15].x < keypoints[16].x

def is_left_wrist_higher(keypoints):
    return keypoints[15].y < keypoints[16].y

def is_left_knee_forward(keypoints):
    return keypoints[25].x < keypoints[26].x

def is_left_wrist_more_left(keypoints):
    return keypoints[15].x < keypoints[16].x

asymmetrical_canonical_side = {
    "Akarna_Dhanurasana": is_left_hip_forward,
    "Bharadvajas_Twist_pose_or_Bharadvajasana_I_": is_left_hip_forward,
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_": is_left_hip_forward,
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_": is_left_hip_forward,
    "Gate_Pose_or_Parighasana_": is_left_hip_forward,
    "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_": is_left_hip_forward,
    "Half_Moon_Pose_or_Ardha_Chandrasana_": is_left_hip_forward,
    "Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_": is_left_hip_forward,
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_": is_left_hip_forward,
    "Lord_of_the_Dance_Pose_or_Natarajasana_": is_left_hip_forward,
    "Low_Lunge_pose_or_Anjaneyasana_": is_left_hip_forward,
    "Noose_Pose_or_Pasasana_": is_left_hip_forward,
    "Pigeon_Pose_or_Kapotasana_": is_left_hip_forward,
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II": is_left_hip_forward,
    "Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_": is_left_hip_forward,
    "Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_": is_left_hip_forward,
    "Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_": is_left_wrist_more_left,
    "Side_Plank_Pose_or_Vasisthasana_": is_left_wrist_higher,
    "Side-Reclining_Leg_Lift_pose_or_Anantasana_": is_left_knee_forward,
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana": is_left_hip_forward,
    "Tree_Pose_or_Vrksasana_": is_left_hip_forward,
    "viparita_virabhadrasana_or_reverse_warrior_pose": is_left_hip_forward,
    "Warrior_I_Pose_or_Virabhadrasana_I_": is_left_hip_forward,
    "Warrior_II_Pose_or_Virabhadrasana_II_": is_left_hip_forward,
    "Warrior_III_Pose_or_Virabhadrasana_III_": is_left_hip_forward,
    "Wild_Thing_pose_or_Camatkarasana_": is_left_wrist_more_left,
    "Rajakapotasana": is_left_hip_forward,
}