import numpy as np
import pandas as pd

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

def scale_keypoints(arr, scale_range):
    """
    Scale each keypoint by Uniform(-scale_range, scale_range)
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :param scale_range: Amount of scaling to apply
    :type scale_range: float
    :return: Array of keypoints with scaled values
    :rtype: numpy.ndarray of shape (33, 4)
    """
    scale = np.random.uniform(1-scale_range, 1+scale_range)
    ctr = np.mean(arr[:, :2], axis=0) # (mean_x, mean_y)
    arr[:, :2] = ((arr[:, :2] - ctr) * scale) + ctr # make the ctr origin, scale, then shift back
    return arr

def translate_keypoints(arr, trans_range):
    """
    Translate each keypoint by Uniform(-trans_range, trans_range)
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :param trans_range: Amount of transformation to apply
    :type trans_range: float
    :return: Array of keypoints with transformed values
    :rtype: numpy.ndarray of shape (33, 4)
    """
    dx, dy = np.random.uniform(-trans_range, trans_range, 2)
    arr[:,0] += dx
    arr[:,1] += dy
    return arr

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
    arr[:, :2] = ((arr[:, :2] - ctr) @ rotation_matrix) + ctr
    return arr

def dropout_keypoints(arr, dropout_prob):
    """
    Drop each keypoint with probability dropout_prob to simulate missing keypoints
    :param arr: Array of keypoints
    :type arr: numpy.ndarray of shape (33, 4)
    :param dropout_prob: Probability of dropout
    :type dropout_prob: float
    :return: Array of keypoints with dropout applied
    :rtype: numpy.ndarray of shape (33, 4)
    """
    drops = np.random.rand(33) < dropout_prob
    arr[drops] = 0
    return arr

x_train = pd.read_parquet('data/processed/x_train.parquet')
y_train = pd.read_parquet('data/processed/y_train.parquet')

rows_aug = []
labels_aug = []
for i in range(len(x_train)):
    arr = x_train.iloc[i].values.reshape((33, 4))
    label = y_train.iloc[i]

    for j in range(3):
        jitter_keypoints(arr, 0.01)
        scale_keypoints(arr, 0.05)
        translate_keypoints(arr, 0.05)
        rotate_keypoints(arr, 10)
        dropout_keypoints(arr, 0.05)
        rows_aug.append(arr.flatten().tolist())
        labels_aug.append(label)

x_aug = pd.DataFrame(rows_aug, columns=x_train.columns)
y_aug = pd.Series(labels_aug, name='label')
x_train_aug = pd.concat([x_train, x_aug], ignore_index=True, axis=0)
y_train_aug = pd.concat([y_train, y_aug], ignore_index=True, axis=0)

x_train_aug.to_parquet('data/processed/x_train_aug.parquet')
pd.DataFrame(y_train_aug, columns=['label']).to_parquet('data/processed/y_train_aug.parquet')