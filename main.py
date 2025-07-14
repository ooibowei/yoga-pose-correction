import cv2
from TTS.api import TTS
import uuid
import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.correction import generate_pose_corrections, remove_angle_from_correction
from app.visualiser import visualise_pose_corrections
from app.predictor import predict_pose
from scripts.utils import generate_keypoints, normalise_keypoints

app = Flask(__name__)

# Initialise models
model_path = 'models/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

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

def generate_tts_audio(text):
    filename = f"temp_{uuid.uuid4().hex}.wav"
    tts.tts_to_file(text=text, file_path=filename)
    with open(filename, 'rb') as file:
        audio_bytes = file.read()
    os.remove(filename)
    return audio_bytes

@app.route('/image', methods=['POST'])
def pose_correction_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pose = request.form.get('pose', None)

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
    img_encode = cv2.imencode(".jpg", annotated_image)[1]
    img_encode_base64 = base64.b64encode(img_encode).decode("utf-8")
    audio_bytes = generate_tts_audio(corrections_text)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return jsonify({
        "annotated_image_base64": img_encode_base64,
        "corrections_audio_base64": audio_base64
    })

@app.route('/video', methods=['POST'])
def pose_correction_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pose = request.form.get('pose', None)

    video_path = f"temp_{uuid.uuid4().hex}.mp4"
    file.save(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = f"output_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    warmup_frames = 30
    frame_count = 0
    corrections_texts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count > warmup_frames:
            annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
            corrections_texts.append(corrections_text)
        else:
            annotated_image = frame
        out.write(annotated_image)
    cap.release()
    out.release()
    os.remove(video_path)

    video = open(output_path, 'rb')
    video_encode_base64 = base64.b64encode(video.read()).decode("utf-8")
    audio_bytes = generate_tts_audio([corrections_text])
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return jsonify({
        "annotated_image_base64": video_encode_base64,
        "corrections_audio_base64": audio_base64
    })


if __name__ == "__main__":
    app.run()

"""
def main(source, pose):
    speech_queue = queue.Queue(maxsize=1)
    threading.Thread(target=speech_worker, args=(speech_queue,), daemon=True).start()

    if source == 'image':
        image_path = 'data/warrior2.jpg'
        image = cv2.imread(image_path)
        annotated_image, corrections_text = process_frame(image, pose_landmarker, pose)

        if not speech_queue.full():
            speech_queue.put(corrections_text)
        cv2.imshow("Pose Correction", annotated_image)
        cv2.waitKey(0)

    elif source == 'video':
        cap = cv2.VideoCapture("data/warrior2.mp4")
        warmup_frames = 30
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > warmup_frames:
                annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
                if not speech_queue.full():
                    speech_queue.put(corrections_text)
            else:
                annotated_image = frame
            cv2.imshow("Pose Correction Video", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif source == 'webcam':
        cap = cv2.VideoCapture(0)
        warmup_frames = 30
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > warmup_frames:
                annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
                if not speech_queue.full():
                    speech_queue.put(corrections_text)
            else:
                annotated_image = frame
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
"""