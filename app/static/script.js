document.addEventListener("DOMContentLoaded", function(){
    const inputType = document.getElementById("input-type");
    const uploadSection = document.getElementById("upload-section");
    const webcamSection = document.getElementById("webcam-section");
    const fileInput = document.getElementById("file-input");
    const uploadButton = document.getElementById("upload-button");
    const webcamStream = document.getElementById("webcam-stream");
    const resultSection = document.getElementById("result-section");

    // Update UI based on input type
    function updateUI() {
        const type = inputType.value;
        if (type === "image" || type === "video") {
            uploadSection.style.display = "block";
            webcamSection.style.display = "none";

            if (type === "image") {
                fileInput.accept = "image/*";
            } else {
                fileInput.accept = "video/*";
            }
        } else if (type === "webcam") {
            uploadSection.style.display = "none";
            webcamSection.style.display = "block";
            startWebcamStream();
        } else {
            uploadSection.style.display = "none";
            webcamSection.style.display = "none";
        }
    }
    inputType.addEventListener("change", updateUI);
    updateUI();

    // Upload button
    uploadButton.addEventListener("click", async (event) => {
        event.preventDefault();
        const type = inputType.value;
        const file = fileInput.files[0];
        if (!file) {
            alert("Please select a file.");
            return;
        }

        uploadButton.disabled = true;
        resultSection.innerHTML = "";
        resultSection.innerHTML = '<span class="spinner"></span> Analysing...';

        const formData = new FormData();
        formData.append("file", file);
        const endpoint = type === "image" ? "/image" : "/video";
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (type === "image") {
            const img = document.createElement("img");
            img.src = "data:image/jpeg;base64," + data.annotated_image_base64;
            resultSection.innerHTML = "";
            resultSection.appendChild(img);
        } else if (type === "video") {
            const video = document.createElement("video");
            video.controls = true
            video.playsInline = true;
            video.src = "data:video/mp4;base64," + data.annotated_video_base64;

            resultSection.innerHTML = "";
            resultSection.appendChild(video);
        }

        uploadButton.disabled = false
    });

    // Webcam video and audio
    function startWebcamStream() {
        webcamStream.src = "/webcam"

        const audioSource = new EventSource("/webcam_corrections"); // audio endpoint
        let audioPlaying = false;
        const audio = new Audio()

        audio.onended = () => {
            audioPlaying = false;
        };
        audioSource.onmessage = (event) => {
            if (!audioPlaying) {
                const data = JSON.parse(event.data);
                audio.src = "data:audio/wav;base64," + data.corrections_audio;
                audio.play();
                audioPlaying = true;
            }
        };
        audioSource.onerror = (err) => {
            console.error("SSE error:", err);
            audioSource.close();
        };
    }
});