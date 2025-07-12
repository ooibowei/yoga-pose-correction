import joblib
import re
from scripts.utils import extract_joint_angles, pose_to_rule, flip_pose, joint_instruction_templates, flipped_joints

pose_target_key_angles = joblib.load('models/pose_target_key_angles.joblib')

def generate_pose_corrections(keypoints_norm, target_pose, threshold=10):
    flipped = False
    if target_pose in pose_to_rule and pose_to_rule[target_pose](keypoints_norm) == False:
        flipped = True
        keypoints_norm = flip_pose(keypoints_norm)
    target_key_joint_angles = pose_target_key_angles[target_pose]
    joint_angles = extract_joint_angles(keypoints_norm)

    corrections = {}
    for joint, target_angle in target_key_joint_angles.items():
        observed_angle = joint_angles[joint]
        if observed_angle is None:
            continue
        diff = observed_angle - target_angle
        if abs(diff) < threshold:
            continue
        direction = 'increase' if diff < 0 else 'decrease'
        if flipped == True and joint in flipped_joints:
            true_joint = flipped_joints[joint]
            instruction = f'{joint_instruction_templates[true_joint][direction]} ({abs(int(diff))} deg)'
        else:
            true_joint = joint
            instruction = f'{joint_instruction_templates[true_joint][direction]} ({abs(int(diff))} deg)'
        corrections[true_joint] = instruction
    return corrections

def remove_angle_from_correction(correction):
    return re.sub(r'\s*\([^)]*\)', '', correction).strip()