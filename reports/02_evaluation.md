# Yoga Pose Correction - Evaluation

We evaluate the final FFN model on the test set

---

## F1 Scores
- The macro F1 score on the test set = 0.866
- Plotting the 10 lowest F1 scores for individual classes, we found that poses like Noose Pose and Tortoise Pose had the lowest F1 scores

![lowest_f1_poses](images/lowest_f1_poses.png)

---

## Misclassified Pairs
We analyse the most common (after normalisation) misclassified pairs of poses. The two most common misclassifications are:
- Noose Pose misclassified as Garland Pose
- Dolphin Pose misclassified as Downward Facing Dog Pose

![misclassified_poses](images/misclassified_poses.png)

We plot sample images of the two common misclassifications. We observe that each pair of poses share a large amount of similarities. When certain body landmarks are obscured, one can potentially be misclassified as the other

![noose_garland](images/noose_garland.png)

![dolphin_downwarddog](images/dolphin_downwarddog.png)