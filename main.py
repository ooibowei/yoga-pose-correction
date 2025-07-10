import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.correction import generate_pose_corrections
from app.visualiser import visualise_pose_corrections
from app.predictor import predict_pose
from scripts.utils import generate_keypoints, normalise_keypoints

def main():
    # Adjust for webcam
    # Add audio, including 1.5s cooldown

    model_path = 'models/pose_landmarker_lite.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    image_path = 'data/6388f2a3b4290800185ccda7.jpg'
    image = cv2.imread(image_path)
    keypoints = generate_keypoints(image, pose_landmarker)
    keypoints_norm = normalise_keypoints(keypoints)
    target_pose = predict_pose(keypoints_norm)

    corrections = generate_pose_corrections(keypoints_norm, target_pose, threshold=10)
    annotated_image = visualise_pose_corrections(image, keypoints, corrections, target_pose)
    cv2.imshow("Pose Correction", annotated_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
