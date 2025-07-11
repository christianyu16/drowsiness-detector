import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load model
@st.cache_resource
def load_model():
    return YOLO("Yolov11best.pt") 

model = load_model()

st.title("Drowsiness Detection from Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_video is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()
    video_path = tfile.name

    # Show original video
    st.video(video_path)
    st.info("Processing video...")

    # Load video and get properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video path
    output_path = os.path.join(tempfile.gettempdir(), "output_drowsy.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(frame, conf=0.3, verbose=False)
        result_frame = results[0].plot()

        out.write(result_frame)

    cap.release()
    out.release()

    st.success("Video processing complete!")
    st.video(output_path)  
