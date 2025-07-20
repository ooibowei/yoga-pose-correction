import cv2
import base64
import os
import json
import tempfile
import numpy as np
from TTS.api import TTS
from queue import Queue
from flask import Flask, request, render_template, jsonify, Response
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.correction import generate_pose_corrections, remove_angle_from_correction
from app.visualiser import visualise_pose_corrections
from app.predictor import predict_pose
from scripts.utils import generate_keypoints, normalise_keypoints

app = Flask(__name__, static_folder='static', template_folder='templates')

model_path = 'models/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)
corrections_queue = Queue()

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
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    tts.tts_to_file(text=text, file_path=temp_path)
    with open(temp_path, 'rb') as file:
        audio_bytes = file.read()
    os.remove(temp_path)
    return audio_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def pose_correction_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pose = request.form.get('pose', None)

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
        img_encode = cv2.imencode(".jpg", annotated_image)[1]
        img_encode_base64 = base64.b64encode(img_encode).decode("utf-8")
        return jsonify({"annotated_image_base64": img_encode_base64})
    except Exception as e:
        return jsonify({"error": str(e)})

def gen_webcam(pose_landmarker, pose=None):
    cap = cv2.VideoCapture(0)
    last_corrections_text = None
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
        if corrections_text != last_corrections_text:
            audio_bytes = generate_tts_audio(corrections_text)
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            corrections_queue.put({"corrections_text": corrections_text, "corrections_audio": audio_base64})
            last_corrections_text = corrections_text
        img_encode = cv2.imencode(".jpg", annotated_image)[1]
        image_bytes = img_encode.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
    cap.release()

@app.route('/webcam', methods=['GET'])
def pose_correction_webcam():
    pose = request.args.get('pose', None)
    try:
        stream = gen_webcam(pose_landmarker, pose)
        if stream is None:
            return jsonify({'error': 'No webcam stream'})
        else:
            return app.response_class(stream, mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/webcam_corrections', methods=['GET'])
def pose_correction_webcam_audio():
    def event_stream():
        while True:
            if not corrections_queue.empty():
                correction = corrections_queue.get()
                yield f"data: {json.dumps(correction)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/video', methods=['POST'])
def pose_correction_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    pose = request.form.get('pose', None)

    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = temp_file.name
        temp_file.close()
        file.save(video_path)
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_file.name
        output_file.close()
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        warmup_frames = 30
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > warmup_frames:
                annotated_image, corrections_text = process_frame(frame, pose_landmarker, pose)
            else:
                annotated_image = frame
            out.write(annotated_image)
        cap.release()
        out.release()
        os.remove(video_path)

        with open(output_path, 'rb') as video:
            video_encode_base64 = base64.b64encode(video.read()).decode("utf-8")
        os.remove(output_path)

        return jsonify({"annotated_video_base64": video_encode_base64})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    from waitress import serve
    serve(app)
