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
    left_hip_pos = arr[23, :]
    left_torso_length = np.linalg.norm(arr[23, :2] - arr[11, :2])
    arr[:, :3] -= left_hip_pos[:3]
    arr[:, :3] /= (left_torso_length + 1e-6)
    return arr

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
    keypoints_arr_norm = normalise_keypoints(keypoints_arr)
    return keypoints_arr_norm

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
    :param keypoints_arr_norm:
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