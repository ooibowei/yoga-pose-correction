import cv2
import argparse
import asyncio
import pyttsx3
import subprocess
import threading
import queue
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.correction import generate_pose_corrections, remove_angle_from_correction
from app.visualiser import visualise_pose_corrections
from app.predictor import predict_pose
from scripts.utils import generate_keypoints, normalise_keypoints

def init_pose_landmarker():
    model_path = 'models/pose_landmarker_lite.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return pose_landmarker

def process_frame(frame, pose_landmarker, pose):
    keypoints = generate_keypoints(frame, pose_landmarker)
    if keypoints is None:
        return frame, "No keypoints detected"
    keypoints_norm = normalise_keypoints(keypoints)
    if pose is None:
        target_pose, target_prob = predict_pose(keypoints_norm)
    else:
        target_pose, target_prob = pose, 1

    corrections = generate_pose_corrections(keypoints_norm, target_pose, threshold=10)
    if corrections:
        corrections_text = ". ".join(remove_angle_from_correction(correction) for correction in corrections.values())
    else:
        corrections_text = "No corrections needed"
    annotated_image = visualise_pose_corrections(frame.copy(), keypoints, corrections, target_pose, target_prob)
    return annotated_image, corrections_text

def speak_text(text):
    subprocess.run(['espeak', text])

def speech_worker(speech_queue):
    last_text = None
    while True:
        text = speech_queue.get()
        if text != last_text and text:
            speak_text(text)
            last_text = text
        speech_queue.task_done()

def main(source, pose):
    pose_landmarker = init_pose_landmarker()
    speech_queue = queue.Queue(maxsize=1)
    threading.Thread(target=speech_worker, args=(speech_queue,), daemon=True).start()

    if source == 'image':
        image_path = 'data/6388f2a3b4290800185ccda7.jpg'
        image = cv2.imread(image_path)
        annotated_image, corrections_text = process_frame(image, pose_landmarker, pose)

        if not speech_queue.full():
            speech_queue.put(corrections_text)
        cv2.imshow("Pose Correction", annotated_image)
        cv2.waitKey(0)

    elif source == 'video':
        cap = cv2.VideoCapture("data/warrior2.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)

            if not speech_queue.full():
                speech_queue.put(corrections_text)
            cv2.imshow("Pose Correction Video", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif source == 'webcam':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)

            if not speech_queue.full():
                speech_queue.put(corrections_text)
            cv2.imshow("Pose Correction Video", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['image', 'video', 'webcam'], default='image')
    parser.add_argument('--pose', type=str, default=None)
    args = parser.parse_args()
    main(args.source, args.pose)