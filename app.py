import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("Yolov11best.pt") 

model = load_model()

st.title("🧠 Drowsiness Detection from Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_video is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)
    st.info("⏳ Running detection on video...")

    # Capture video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0 or width == 0 or height == 0:
        st.error("❌ Invalid video file.")
    else:
        output_path = os.path.join(tempfile.gettempdir(), "processed_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.3, verbose=False)
            result_frame = results[0].plot()

            out.write(result_frame)

        cap.release()
        out.release()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success("✅ Video processed successfully!")
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Processed Video", f, file_name="drowsiness_result.avi")
        else:
            st.error("❌ Failed to process video.")
