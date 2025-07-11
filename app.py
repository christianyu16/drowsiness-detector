import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# Cache model for performance
@st.cache_resource
def load_model():
    return YOLO("Yolov11best.pt")  # Ensure this file exists in the same directory or use full path

model = load_model()

st.title("Drowsiness Detection from Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_video is not None:
    # Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()
    video_path = tfile.name

    st.video(video_path)
    st.info("Running detection on video (this may take a while)...")

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video file
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model.predict(frame, conf=0.3, verbose=False)
        result_frame = results[0].plot()

        # Write processed frame
        out.write(result_frame)

    cap.release()
    out.release()

    st.success("Video processing complete!")

    # Display processed video
    with open(output_path, 'rb') as processed_file:
        video_bytes = processed_file.read()
        st.video(video_bytes)
