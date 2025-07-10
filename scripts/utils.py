import numpy as np
import mediapipe as mp
import cv2

def normalise_keypoints(arr):
    """
    Normalise keypoint location and scale based on a reference landmark
    Location is normalised based on position of left hip (index 23)
    Scale is normalised based on length of left torso, which is the distance between the left hip (index 23) and the left shoulder (index 11)
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :return: Array of normalised keypoints
    :rtype: numpy.ndarray of shape (33, 4)
    """
    arr_norm = arr.copy()
    left_hip_pos = arr_norm[23, :]
    left_torso_length = np.linalg.norm(arr_norm[23, :2] - arr_norm[11, :2])
    arr_norm[:, :3] -= left_hip_pos[:3]
    arr_norm[:, :3] /= (left_torso_length + 1e-6)
    return arr_norm

def generate_keypoints(image, pose_landmarker):
    """
    Generate normalised keypoints from image using MediaPipe
    :param image: Image file (jpg/jpeg/png) read by cv2 in BGR
    :type image: numpy.ndarray of shape (height, width, 3)
    :param pose_landmarker: Object from vision.PoseLandmarker, specifying the type of MediaPipe model to use
    :type pose_landmarker: Object from vision.PoseLandmarker.create_from_options
    :return: Array of keypoints from image
    :rtype: numpy.ndarray of shape (33, 4)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    pose_res = pose_landmarker.detect(mp_image)
    if pose_res.pose_landmarks is None or len(pose_res.pose_landmarks) == 0:
        return None
    keypoints_arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]])
    return keypoints_arr

def calculate_angle(a, b, c, vis_threshold=0):
    """
    Calculate angle between three keypoints (a, b, c)
    :param a/b/c: keypoints each of the form (x, y, z, visibility)
    :type a/b/c: numpy.ndarray of shape (1, 4)
    :param vis_threshold: threshold for visualisation. Drop the joint angle if visibility <= vis_threshold
    :type vis_threshold: float
    :return: Angle at joint b
    :rtype: float
    """
    if a[3] <= vis_threshold or b[3] < vis_threshold or c[3] <= vis_threshold:
        return None
    a = a[:2]
    b = b[:2]
    c = c[:2]
    bc = c-b
    ba = a-b
    if (np.linalg.norm(ba) * np.linalg.norm(bc)) == 0:
        return None
    deg_radian = np.arccos(np.dot(bc, ba) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    deg = np.degrees(deg_radian)
    return deg

def calculate_horizontal_ref_angle(a, b, vis_threshold=0):
    """
    Calculate angle between line segment (two keypoints a, b) and horizontal x-axis. Checks if a body part is tilted
    :param a/b: keypoints each of the form (x, y, z, visibility)
    :type a/b: numpy.ndarray of shape (1, 4)
    :param vis_threshold: threshold for visualisation. Drop the horizontal angle if visibility <= vis_threshold
    :type vis_threshold: float
    :return: Angle of tilt relative to horizontal axis
    :rtype: float
    """
    if a[3] <= vis_threshold or b[3] < vis_threshold:
        return None
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return np.degrees(np.arctan2(dy, dx))

def extract_joint_angles(keypoints_arr_norm, vis_threshold=0):
    """
    Extract a dictionary of pre-defined joint angles
    :param keypoints_arr_norm: Normalised keypoints from image
    :type keypoints_arr_norm: numpy.ndarray of shape (33, 4)
    :return: Dictionary of pre-defined joint angles
    :rtype: dict
    """
    angles = {}
    angles['left_elbow'] = calculate_angle(keypoints_arr_norm[11], keypoints_arr_norm[13], keypoints_arr_norm[15], vis_threshold)
    angles['right_elbow'] = calculate_angle(keypoints_arr_norm[12], keypoints_arr_norm[14], keypoints_arr_norm[16], vis_threshold)
    angles['left_shoulder'] = calculate_angle(keypoints_arr_norm[23], keypoints_arr_norm[11], keypoints_arr_norm[13], vis_threshold)
    angles['right_shoulder'] = calculate_angle(keypoints_arr_norm[24], keypoints_arr_norm[12], keypoints_arr_norm[14], vis_threshold)
    angles['left_hip'] = calculate_angle(keypoints_arr_norm[11], keypoints_arr_norm[23], keypoints_arr_norm[25], vis_threshold)
    angles['right_hip'] = calculate_angle(keypoints_arr_norm[12], keypoints_arr_norm[24], keypoints_arr_norm[26], vis_threshold)
    angles['left_knee'] = calculate_angle(keypoints_arr_norm[23], keypoints_arr_norm[25], keypoints_arr_norm[27], vis_threshold)
    angles['right_knee'] = calculate_angle(keypoints_arr_norm[24], keypoints_arr_norm[26], keypoints_arr_norm[28], vis_threshold)
    angles['torso_bend'] = calculate_angle(keypoints_arr_norm[11], keypoints_arr_norm[23], keypoints_arr_norm[24], vis_threshold)
    angles['head_tilt'] = calculate_angle(keypoints_arr_norm[0], keypoints_arr_norm[11], keypoints_arr_norm[12], vis_threshold)
    angles['shoulder_tilt'] = calculate_horizontal_ref_angle(keypoints_arr_norm[11], keypoints_arr_norm[12], vis_threshold)
    angles['hip_tilt'] = calculate_horizontal_ref_angle(keypoints_arr_norm[23], keypoints_arr_norm[24], vis_threshold)
    return {k: v for k, v in angles.items() if v is not None}


def is_left_knee_more_bent(keypoints):
    left_angle = calculate_angle(keypoints[23], keypoints[25], keypoints[27])
    right_angle = calculate_angle(keypoints[24], keypoints[26], keypoints[28])
    return left_angle < right_angle

def is_left_leg_lifted(keypoints):
    return keypoints[25][1] < keypoints[26][1]

def is_left_wrist_lower(keypoints):
    return keypoints[15][1] > keypoints[16][1]

def is_left_knee_forward(keypoints):
    return keypoints[25][2] < keypoints[26][2]

def is_left_shoulder_lower(keypoints):
    return keypoints[11][1] > keypoints[12][1]

def is_left_hip_lower(keypoints):
    return keypoints[23][1] > keypoints[24][1]

def is_left_ankle_lower(keypoints):
    return keypoints[27][1] > keypoints[28][1]

def flip_pose(keypoints):
    """
    Swap left and right keypoint labels
    :param keypoints: keypoints from image
    :type keypoints: numpy.ndarray of shape (33, 4)
    :return: Keypoints array with left and right labels swapped
    :rtype: numpy.ndarray of shape (33, 4)
    """
    left_indices = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_indices = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    arr = keypoints.copy()
    arr[:,0] = -keypoints[:,0]
    arr[left_indices,:] = keypoints[right_indices,:].copy()
    arr[right_indices,:] =  keypoints[left_indices,:].copy()
    return arr

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
    "Legs-Up-the-Wall_Pose_or_Viparita_Karani_",
    "Locust_Pose_or_Salabhasana_",
    "Peacock_Pose_or_Mayurasana_",
    "Pigeon_Pose_or_Kapotasana_",
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
    "Heron_Pose_or_Krounchasana_",
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_",
    "Lord_of_the_Dance_Pose_or_Natarajasana_",
    "Low_Lunge_pose_or_Anjaneyasana_",
    "Noose_Pose_or_Pasasana_",
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

all_poses = symmetrical_poses.union(asymmetrical_poses)

pose_to_rule = {
    "Akarna_Dhanurasana": is_left_leg_lifted,
    "Bharadvajas_Twist_pose_or_Bharadvajasana_I_": is_left_knee_forward,
    "Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_": is_left_knee_more_bent,
    "Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_": is_left_wrist_lower,
    "Gate_Pose_or_Parighasana_": is_left_leg_lifted,
    "Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana_": is_left_leg_lifted,
    "Half_Moon_Pose_or_Ardha_Chandrasana_": is_left_leg_lifted,
    "Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana_": is_left_knee_more_bent,
    "Heron_Pose_or_Krounchasana_": is_left_leg_lifted,
    "Intense_Side_Stretch_Pose_or_Parsvottanasana_": is_left_knee_more_bent,
    "Lord_of_the_Dance_Pose_or_Natarajasana_": is_left_leg_lifted,
    "Low_Lunge_pose_or_Anjaneyasana_": is_left_knee_more_bent,
    "Noose_Pose_or_Pasasana_": is_left_shoulder_lower,
    "Pose_Dedicated_to_the_Sage_Koundinya_or_Eka_Pada_Koundinyanasana_I_and_II": is_left_hip_lower,
    "Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana_": is_left_leg_lifted,
    "Revolved_Head-to-Knee_Pose_or_Parivrtta_Janu_Sirsasana_": is_left_wrist_lower,
    "Side_Crane_(Crow)_Pose_or_Parsva_Bakasana_": is_left_wrist_lower,
    "Side_Plank_Pose_or_Vasisthasana_": is_left_wrist_lower,
    "Side-Reclining_Leg_Lift_pose_or_Anantasana_": is_left_leg_lifted,
    "Standing_big_toe_hold_pose_or_Utthita_Padangusthasana": is_left_leg_lifted,
    "Tree_Pose_or_Vrksasana_": is_left_leg_lifted,
    "viparita_virabhadrasana_or_reverse_warrior_pose": is_left_knee_more_bent,
    "Warrior_I_Pose_or_Virabhadrasana_I_": is_left_knee_more_bent,
    "Warrior_II_Pose_or_Virabhadrasana_II_": is_left_knee_more_bent,
    "Warrior_III_Pose_or_Virabhadrasana_III_": is_left_leg_lifted ,
    "Wild_Thing_pose_or_Camatkarasana_": is_left_wrist_lower,
    "Rajakapotasana": is_left_ankle_lower
}

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
    "Warrior_II_Pose_or_Virabhadrasana_II_": ["left_knee", "left_hip", "right_knee", "left_shoulder", "right_shoulder", "torso_bend"],
    "Warrior_III_Pose_or_Virabhadrasana_III_": ["left_knee", "right_hip", "right_knee", "hip_tilt"],
    "Wild_Thing_pose_or_Camatkarasana_": ["left_knee", "right_knee", "torso_bend"],
    "Rajakapotasana": ["left_knee", "right_knee", "hip_tilt"]
}

flipped_joints = {
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
}

joint_instruction_templates = {
    "left_elbow": {
        "increase": "Straighten left elbow",
        "decrease": "Bend left elbow"
    },
    "right_elbow": {
        "increase": "Straighten right elbow",
        "decrease": "Bend right elbow"
    },
    "left_shoulder": {
        "increase": "Raise left arm",
        "decrease": "Lower left arm"
    },
    "right_shoulder": {
        "increase": "Raise right arm",
        "decrease": "Lower right arm"
    },
    "left_hip": {
        "increase": "Lift torso or reduce left hip bend",
        "decrease": "Sink into left hip"
    },
    "right_hip": {
        "increase": "Lift torso or reduce right hip bend",
        "decrease": "Sink into right hip"
    },
    "left_knee": {
        "increase": "Straighten left knee",
        "decrease": "Bend left knee"
    },
    "right_knee": {
        "increase": "Straighten right knee",
        "decrease": "Bend right knee"
    },
    "torso_bend": {
        "increase": "Straighten torso",
        "decrease": "Lean torso"
    },
    "head_tilt": {
        "increase": "Straighten head",
        "decrease": "Tilt head"
    },
    "shoulder_tilt": {
        "increase": "Raise left shoulder or lower right shoulder",
        "decrease": "Raise right shoulder or lower left shoulder"
    },
    "hip_tilt": {
        "increase": "Raise left hip or lower right hip",
        "decrease": "Raise right hip or lower left hip"
    }
}

pose_connections = [
    (0, 11), (0, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (23, 24),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31), (28, 32)
]

joint_to_index = {
    "left_elbow": 13,
    "right_elbow": 14,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_wrist": 15,
    "right_wrist": 16,
    "torso_bend": 24,
    "shoulder_tilt": 11,
    "hip_tilt": 23,
    "head_tilt": 0,
}
