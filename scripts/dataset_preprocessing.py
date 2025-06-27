import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True)

image_path = 'data/raw/Akarna_Dhanurasana/Akarna_Dhanurasana_image_1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
res = pose_image.process(image_rgb)

keypoints = [] # (x1, y1, z1, v1, x2, y2, z2,...)
for lm in res.pose_landmarks.landmark:
    keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

mp_drawing = mp.solutions.drawing_utils
mp_drawing.draw_landmarks(
    image,
    res.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
)
for idx, landmark in enumerate(res.pose_landmarks.landmark):
    h, w, _ = image.shape
    x, y = int(landmark.x * w), int(landmark.y * h)

    cv2.putText(
        image,
        f'{mp_pose.PoseLandmark(idx).name}',
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
cv2.imshow('Pose Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()