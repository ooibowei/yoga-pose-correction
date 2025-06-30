import numpy as np

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