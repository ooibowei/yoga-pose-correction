import cv2
from scripts.utils import pose_connections, joint_to_index

def visualise_pose_corrections(image, keypoints, corrections, target_pose, target_prob):
    annotated_image = image.copy()
    h, w = annotated_image.shape[:2]

    for i, j in pose_connections:
        if keypoints[i][3] > 0.5 and keypoints[j][3] > 0.5:
            pt_i = (int(keypoints[i][0] * w), int(keypoints[i][1] * h))
            pt_j = (int(keypoints[j][0] * w), int(keypoints[j][1] * h))
            cv2.line(annotated_image, pt_i, pt_j, (200, 200, 200), 2)

    correction_indices = {joint_to_index[joint] for joint in corrections}
    for i, (x, y, z, vis) in enumerate(keypoints):
        if vis > 0.5 and (i== 0 or i >= 11):
            circle_x, circle_y = int(x * w), int(y * h)
            if i in correction_indices:
                colour = (0, 0, 255)
            else:
                colour = (0, 255, 0)
            cv2.circle(annotated_image, (circle_x, circle_y), 4, colour, -1)

    box_x, box_y, box_w = 10, 10, 480
    line_height = 17
    padding_y = 10
    n_lines = len(corrections) if corrections else 1
    box_h = padding_y * 2 + 30 + line_height * n_lines
    cv2.rectangle(annotated_image, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
    cv2.rectangle(annotated_image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.putText(annotated_image, f"Pose: {target_pose.replace('_', ' ')} (Prob {target_prob:.2f})", (box_x + 10, box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    if not corrections:
        cv2.putText(annotated_image, "- No corrections needed", (box_x + 10, box_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    else:
        for i, text in enumerate(corrections.values()):
            y_pos = box_y + 50 + i * line_height
            cv2.putText(annotated_image, f"- {text}", (box_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return annotated_image