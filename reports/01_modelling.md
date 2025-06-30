# Yoga Pose Correction - Modelling

We XX 

---

## Data Augmentation

To improve model generalisability and robustness, we augmented the training set by applying the following procedure 3 separate times to each training set pose. These augmentations reflect real-world variations like different camera angles and pose imperfections
- Added small random Gaussian noise (jitter) drawn from a N(0, 0.0001)
- Rotated around the center by -10 to 10 degrees
- Randomly dropped landmarks with probability 0.05
 