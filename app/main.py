import cv2
import base64
import os
import json
import tempfile
import time
import asyncio
import aiofiles
import threading
import numpy as np
from edge_tts import Communicate
from fastapi import FastAPI, UploadFile, Request, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.correction import generate_pose_corrections, remove_angle_from_correction
from app.visualiser import visualise_pose_corrections
from app.predictor import predict_pose
from scripts.utils import generate_keypoints, normalise_keypoints, generate_keypoints_async

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

latest_result = {"image": None, "keypoints": None}
result_lock = threading.Lock()
corrections_queue = asyncio.Queue()
model_path = 'models/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result["image"] = output_image
        latest_result["keypoints"] = result.pose_landmarks
options_livestream = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=result_callback)
options_video = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
options_image = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
pose_landmarker_livestream = vision.PoseLandmarker.create_from_options(options_livestream)
pose_landmarker_video = vision.PoseLandmarker.create_from_options(options_video)
pose_landmarker_image = vision.PoseLandmarker.create_from_options(options_image)

async def generate_tts_audio_async(text):
    temp_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    communicate = Communicate(text)
    await communicate.save(temp_path)
    async with aiofiles.open(temp_path, "rb") as f:
        audio = await f.read()
    os.remove(temp_path)
    return audio

async def tts_background_task(text):
    audio_bytes = await generate_tts_audio_async(text)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    await corrections_queue.put({"corrections_text": text, "corrections_audio": audio_base64})

def process_frame(frame, pose_landmarker, pose=None, timestamp_ms=0):
    keypoints = generate_keypoints(frame, pose_landmarker, timestamp_ms)
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

async def gen_webcam_stream(pose_landmarker, pose=None, request=None):
    last_tts_time = 0
    frame_count = 0
    last_encoded_image = None
    cap = cv2.VideoCapture(0)
    last_corrections_text = None
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    while True:
        if await request.is_disconnected():
            print("Client disconnected, stopping webcam stream.")
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            timestamp_ms = int(time.time() * 1000)
            generate_keypoints_async(frame, pose_landmarker, timestamp_ms)
            await asyncio.sleep(0.03)
            with result_lock:
                keypoints = latest_result["keypoints"]
                mp_image = latest_result["image"]
            if keypoints is not None and mp_image is not None:
                if keypoints and hasattr(keypoints[0], 'x'):
                    keypoints_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in keypoints])
                elif keypoints and isinstance(keypoints[0], list) and hasattr(keypoints[0][0], 'x'):
                    keypoints_array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in keypoints[0]])
                else:
                    keypoints_array = None
                if keypoints_array is None:
                    continue
                keypoints_norm = normalise_keypoints(keypoints_array)
                if pose is None:
                    target_pose, target_prob = predict_pose(keypoints_norm)
                else:
                    target_pose, target_prob = pose, 1
                corrections = generate_pose_corrections(keypoints_norm, target_pose, threshold=10)
                if corrections:
                    corrections_text = ". ".join(remove_angle_from_correction(c) for c in corrections.values())
                else:
                    corrections_text = "No corrections needed"
                annotated_image = visualise_pose_corrections(frame.copy(), keypoints_array, corrections, target_pose, target_prob)
                if corrections_text != last_corrections_text:
                    now = time.time()
                    if now - last_tts_time > 5:
                        asyncio.create_task(tts_background_task(corrections_text))
                        last_tts_time = now
                        last_corrections_text = corrections_text
                img_encode = cv2.imencode(".jpg", annotated_image)[1]
                last_encoded_image = img_encode.tobytes()
            if last_encoded_image is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + last_encoded_image + b'\r\n')
    cap.release()

@app.get('/')
async def index():
    return FileResponse('app/templates/index.html')

@app.post('/image')
async def pose_correction_image(file: UploadFile, pose: str = Form(None)):
    try:
        file_content = await file.read()
        frame = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
        annotated_image, corrections_text = process_frame(frame, pose_landmarker_image, pose)
        img_encode = cv2.imencode(".jpg", annotated_image)[1]
        img_encode_base64 = base64.b64encode(img_encode).decode("utf-8")
        return JSONResponse({'annotated_image_base64': img_encode_base64})
    except Exception as e:
        return JSONResponse({'error': str(e)})

@app.post('/video')
async def pose_correction_video(file: UploadFile, pose: str = Form(None)):
    try:
        pose_landmarker_video = vision.PoseLandmarker.create_from_options(options_video)
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        input_path = input_temp.name
        input_temp.close()
        content = await file.read()
        async with aiofiles.open(input_path, "wb") as f:
            await f.write(content)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_filename = f"processed_{int(time.time())}.mp4"
        output_path = f"app/static/processed/{output_filename}"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = int((frame_count / fps) * 1000)
            annotated_image, corrections_text = process_frame(frame, pose_landmarker_video, pose, timestamp_ms)
            out.write(annotated_image)
            frame_count += 1
        cap.release()
        out.release()
        os.remove(input_path)
        return JSONResponse({"video_url": f"/static/processed/{output_filename}"})

    except Exception as e:
        return JSONResponse({'error': str(e)})

@app.delete("/video")
async def delete_video(filename: str = Query(...)):
    try:
        filepath = os.path.join("app/static/processed", filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return {"status": "deleted"}
        else:
            return {"status": "file not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get('/webcam')
async def pose_correction_webcam(request: Request, pose: str = Query(None)):
    try:
        print(f"New /webcam request with pose: {pose}")
        stream = gen_webcam_stream(pose_landmarker_livestream, pose, request)
        if stream is None:
            return JSONResponse(content={'error': 'No webcam stream'})
        else:
            return StreamingResponse(stream, media_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return JSONResponse(content={'error': str(e)})

@app.get('/webcam_corrections')
async def pose_correction_webcam_audio():
    async def event_stream():
        while True:
            correction = await corrections_queue.get()
            yield f"data: {json.dumps(correction)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")