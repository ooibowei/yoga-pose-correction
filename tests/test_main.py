import pytest
import pytest_asyncio
import cv2
import io
import numpy as np
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_pose_correction_image(async_client, mocker):
    mocker.patch("app.main.process_frame", return_value=(np.zeros((480, 640, 3), dtype=np.uint8), "No corrections needed"))
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_encoded = cv2.imencode('.jpg', img)[1]
    dummy_image = io.BytesIO(img_encoded.tobytes())

    files = {"file": ("test.jpg", dummy_image, "image/jpeg")}
    data = {"pose": "Warrior_II_Pose_or_Virabhadrasana_II_"}
    response = await async_client.post("/image", files=files, data=data)

    assert response.status_code == 200
    assert "annotated_image_base64" in response.json()

@pytest.mark.asyncio
async def test_pose_correction_video(async_client, mocker):
    mocker.patch("app.main.process_frame", return_value=(np.zeros((480, 640, 3), dtype=np.uint8), "No corrections needed"))
    dummy_video_content = io.BytesIO(b"fake video content")

    files = {"file": ("test.mp4", dummy_video_content, "video/mp4")}
    data = {"pose": "Warrior_II_Pose_or_Virabhadrasana_II_"}
    response = await async_client.post("/video", files=files, data=data)

    assert response.status_code == 200
    assert "annotated_video_base64" in response.json()

@pytest.mark.asyncio
async def test_webcam_route_simple(async_client, mocker):
    async def dummy_stream(*args, **kwargs):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + np.zeros((10, 10, 3), dtype=np.uint8).tobytes() + b"\r\n"
    mocker.patch("app.main.gen_webcam_stream", side_effect=dummy_stream)

    response = await async_client.get("/webcam")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    async for chunk in response.aiter_bytes():
        assert b"--frame" in chunk
        break