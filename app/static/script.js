document.addEventListener("DOMContentLoaded", function(){
    let audioSource = null;
    const inputType = document.getElementById("input-type");
    const uploadSection = document.getElementById("upload-section");
    const webcamSection = document.getElementById("webcam-section");
    const fileInput = document.getElementById("file-input");
    const poseInput = document.getElementById("pose-input");
    const uploadButton = document.getElementById("upload-button");
    const webcamStream = document.getElementById("webcam-stream");
    const resultSection = document.getElementById("result-section");

    // Update UI based on input type
    function updateUI() {
        const type = inputType.value;
        if (type !== "webcam") {
            webcamStream.src = "";
            if (audioSource) {
                audioSource.close();
                audioSource = null;
            }
        }
        if (type === "image" || type === "video") {
            uploadSection.style.display = "block";
            webcamSection.style.display = "none";
            resultSection.style.display = "block";
            resultSection.innerHTML = "";

            if (type === "image") {
                fileInput.accept = "image/*";
            } else {
                fileInput.accept = "video/*";
            }
        } else if (type === "webcam") {
            uploadSection.style.display = "none";
            webcamSection.style.display = "block";
            resultSection.style.display = "none";
            resultSection.innerHTML = "";
            startWebcamStream();
        } else {
            uploadSection.style.display = "none";
            webcamSection.style.display = "none";
            resultSection.style.display = "none";
            resultSection.innerHTML = "";
        }
    }
    inputType.addEventListener("change", updateUI);
    updateUI();

    poseInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        const type = inputType.value;
        if (type === "webcam") {
            startWebcamStream();
        } else if (type === "image" || type === "video") {
            if (!file) return;
            uploadCurrentFileWithPose();
        }
    });

    // Upload
    async function uploadCurrentFileWithPose() {
        const type = inputType.value;
        const file = fileInput.files[0];
        const pose = poseInput.value;
        if (!file) {
            alert("Please select a file.");
            return;
        }

        uploadButton.disabled = true;
        resultSection.innerHTML = '<span class="spinner"></span> Analysing...';
        const formData = new FormData();
        formData.append("file", file);
        if (pose !== "") {
            formData.append("pose", pose);
        }
        const endpoint = type === "image" ? "/image" : "/video";
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            alert("Error: " + (data.error || "Unknown error"));
            uploadButton.disabled = false;
            return;
        }
        resultSection.innerHTML = "";
        if (type === "image") {
            const img = document.createElement("img");
            img.src = "data:image/jpeg;base64," + data.annotated_image_base64;
            resultSection.appendChild(img);
        } else if (type === "video") {
            const video = document.createElement("video");
            video.controls = true;
            video.playsInline = true;
            video.src = "data:video/mp4;base64," + data.annotated_video_base64;
            resultSection.appendChild(video);
        }
        uploadButton.disabled = false;
    }
    uploadButton.addEventListener("click", async (event) => {
        event.preventDefault();
        uploadCurrentFileWithPose();
    });

    // Webcam video and audio
    function startWebcamStream() {
        if (webcamStream.src) {
            webcamStream.src = "";
        }
        if (audioSource) {
            audioSource.close();
            audioSource = null;
        }

        const pose = poseInput.value;
        if (pose === "") {
            webcamStream.src = "/webcam"
        } else {
            webcamStream.src = `/webcam?pose=${encodeURIComponent(pose)}`;
        }

        audioSource = new EventSource("/webcam_corrections");
        const audio = new Audio()
        let audioPlaying = false;

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