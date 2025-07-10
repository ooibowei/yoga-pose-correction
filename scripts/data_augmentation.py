import numpy as np
import pandas as pd
from scripts.utils import normalise_keypoints

def jitter_keypoints(arr, sigma):
    """
    Apply N(0, sigma2) jitter to each keypoint
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :param sigma: Variance in size of jitter to be applied
    :type sigma: float
    :return: Array of keypoints with jittered values
    :rtype: numpy.ndarray of shape (33, 4)
    """
    noise = np.random.normal(0, sigma, arr.shape)
    return arr + noise

def rotate_keypoints(arr, angle_range):
    """
    Rotate each keypoint by Uniform(-angle_range, angle_range)
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :param angle_range: Amount of rotation to apply
    :type angle_range: float
    :return: Array of keypoints with rotated values
    :rtype: numpy.ndarray of shape (33, 4)
    """
    angle = np.radians(np.random.uniform(-angle_range, angle_range))
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    ctr = np.mean(arr[:, :2], axis=0)

    rotated = arr.copy()
    rotated[:, :2] = ((arr[:, :2] - ctr) @ rotation_matrix) + ctr
    return rotated

x_train = pd.read_parquet('data/processed/x_train.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet')['label']

rows_aug = []
labels_aug = []
for i in range(len(x_train)):
    x = x_train.iloc[i].values.reshape((33, 4))
    label = y_train.iloc[i]

    for j in range(3):
        x_aug = jitter_keypoints(x, 0.01)
        x_aug = rotate_keypoints(x_aug, 10)
        x_aug_norm = normalise_keypoints(x_aug)
        rows_aug.append(x_aug_norm.flatten().tolist())
        labels_aug.append(label)

x_aug = pd.DataFrame(rows_aug, columns=x_train.columns)
y_aug = pd.Series(labels_aug, name='label')
x_train_aug = pd.concat([x_train, x_aug], ignore_index=True, axis=0)
y_train_aug = pd.concat([y_train, y_aug], ignore_index=True, axis=0)

x_train_aug.to_parquet('data/processed/x_train_aug.parquet')
pd.DataFrame(y_train_aug, columns=['label']).to_parquet('data/processed/y_train_aug.parquet')