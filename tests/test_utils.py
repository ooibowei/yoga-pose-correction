import pytest
import numpy as np
from numpy.testing import assert_allclose
from scripts.utils import normalise_keypoints, calculate_angle, extract_joint_angles

def test_normalise_keypoints():
    keypoints = np.zeros((33, 4), dtype=np.float32)
    keypoints[23] = [1.0, 2.0, 3.0, 1.0]
    normed = normalise_keypoints(keypoints)
    assert_allclose(normed[23, :3], [0.0, 0.0, 0.0], rtol=1e-5)

def test_calculate_angle_valid():
    a = np.array([0, 1, 0, 1])
    b = np.array([0, 0, 0, 1])
    c = np.array([1, 0, 0, 1])
    angle = calculate_angle(a, b, c)
    assert np.isclose(angle, 90.0)

def test_extract_joint_angles():
    keypoints = np.random.rand(33, 4).astype(np.float32)
    keypoints[:, 3] = 1.0
    normed = normalise_keypoints(keypoints)
    angles = extract_joint_angles(normed)
    assert isinstance(angles, dict)
    assert "left_knee" in angles
