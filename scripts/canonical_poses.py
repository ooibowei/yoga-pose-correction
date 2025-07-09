from scripts.utils import calculate_angle

# For each asymmetric pose, need to decide if it should be flipped to the canonical side (left)

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

def flip_pose(keypoints):
    left_indices = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_indices = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    arr = keypoints.copy()
    arr[:,0] = -keypoints[:,0]
    arr[left_indices,:] = keypoints[right_indices,:].copy()
    arr[right_indices,:] =  keypoints[left_indices,:].copy()
    return arr