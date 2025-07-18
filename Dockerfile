FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/utils.py ./scripts/utils.py
COPY models/best_model_state.pt ./models/best_model_state.pt
COPY models/label_encoder.joblib ./models/label_encoder.joblib
COPY models/scaler.joblib ./models/scaler.joblib
COPY models/model_metadata.joblib ./models/model_metadata.joblib
COPY models/pose_landmarker_lite.task ./models/pose_landmarker_lite.task
COPY models/pose_target_key_angles.joblib ./models/pose_target_key_angles.joblib

CMD ["python", "-m", "app.main"]