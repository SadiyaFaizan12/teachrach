import streamlit as st
import cv2
import numpy as np
import os
import supervision as sv
from collections import deque
from ultralytics import YOLO

# --- Model Configuration ---
# Load the YOLOv8s model. It is more accurate than the 'n' version.
# It will be downloaded automatically on the first run.
# this is the change i have made
try:
    model = YOLO('yolov8s.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# The COCO dataset (which yolov8s is trained on) labels a sports ball as class 32.
SPORTS_BALL_CLASS_ID = 32

# --- Strike Zone ---
# Default strike zone coordinates, adjustable in the UI
strike_zone_coords = {'x1': 200, 'y1': 150, 'x2': 400, 'y2': 350}

# --- Custom Ball Tracker ---
# This class helps stabilize tracking by focusing on the most likely
# object when multiple detections occur.
class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            self.buffer.clear()
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        if not self.buffer:
            self.buffer.append(xy)

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        self.buffer.append(xy)
        return detections[[index]]

# --- Custom Ball Annotator ---
# This class draws a "comet trail" for the ball's trajectory.
class BallAnnotator:
    def __init__(self, radius: int = 10, buffer_size: int = 30, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('cool', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i <= 1:
            return self.radius
        return int(self.radius * (i / (max_i - 1)))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) > 0:
            xy = detections.get_anchors_coordinates(sv.Position.CENTER).astype(int)
            self.buffer.append(xy)
        else:
            self.buffer.append(np.array([]))

        for i, points in enumerate(self.buffer):
            if points.size == 0:
                continue
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in points:
                if interpolated_radius > 0:
                    cv2.circle(img=frame, center=tuple(center), radius=interpolated_radius, color=color.as_bgr(), thickness=self.thickness)
        return frame

def analyze_video(video_path, sz_coords, confidence_threshold):
    """
    Analyzes the video in two phases using a local YOLOv8 model:
    1. Detection & Tracking: Go through all frames and get ball positions.
    2. Annotation & Rendering: Draw the trajectory and create the output video.
    """
    video_info = sv.VideoInfo.from_video_path(video_path)
    w, h = video_info.width, video_info.height

    # --- Phase 1: Detection and Tracking ---
    st.write("Phase 1: Detecting ball in all frames (using local model)...")
    progress_bar_detect = st.progress(0)
    
    # This callback now uses the local YOLO model for much faster inference.
    def inference_callback(patch: np.ndarray) -> sv.Detections:
        # Pass the user-defined confidence threshold to the model
        results = model(patch, conf=confidence_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    slicer = sv.InferenceSlicer(callback=inference_callback, slice_wh=(w // 2 + 100, h // 2 + 100), overlap_ratio_wh=(0.2, 0.2), iou_threshold=0.1)
    tracker = BallTracker()
    
    all_frames = []
    tracked_detections = []
    
    frame_generator = sv.get_video_frames_generator(video_path)
    for i, frame in enumerate(frame_generator):
        all_frames.append(frame)
        raw_detections = slicer(frame)
        # Filter for the 'sports ball' class ID
        ball_detections = raw_detections[raw_detections.class_id == SPORTS_BALL_CLASS_ID]
        tracked_ball = tracker.update(ball_detections)
        tracked_detections.append(tracked_ball)
        progress_bar_detect.progress((i + 1) / video_info.total_frames)

    # --- Add explicit feedback on detection results ---
    num_detections = sum(1 for d in tracked_detections if len(d) > 0)
    st.info(f"Detection Phase Complete: Found a ball in {num_detections} out of {video_info.total_frames} frames.")
    if num_detections == 0:
        st.warning("Warning: The model did not detect a ball in any frame. No trajectory can be drawn.")
        
    # --- Phase 2: Annotation and Rendering ---
    st.write("Phase 2: Generating output video and graph...")
    progress_bar_render = st.progress(0)

    ball_annotator = BallAnnotator(radius=10, buffer_size=30)
    output_video_path = 'output.mp4'

    with sv.VideoSink(output_video_path, video_info=video_info) as sink:
        for i, frame in enumerate(all_frames):
            detections_for_frame = tracked_detections[i]
            annotated_frame = ball_annotator.annotate(frame.copy(), detections_for_frame)
            cv2.rectangle(annotated_frame, (sz_coords['x1'], sz_coords['y1']), (sz_coords['x2'], sz_coords['y2']), (255, 0, 0), 2)
            sink.write_frame(annotated_frame)
            progress_bar_render.progress((i + 1) / len(all_frames))

    return output_video_path, ball_annotator

def main():
    st.title("Softball Pitch Analyzer 🥎 (Local YOLOv8)")

    st.sidebar.header("Configuration")
    
    # Add a confidence slider to the sidebar
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    
    st.sidebar.markdown("Adjust the strike zone coordinates as needed.")
    strike_zone_coords['x1'] = st.sidebar.number_input("Strike Zone X1", value=200)
    strike_zone_coords['y1'] = st.sidebar.number_input("Strike Zone Y1", value=150)
    strike_zone_coords['x2'] = st.sidebar.number_input("Strike Zone X2", value=400)
    strike_zone_coords['y2'] = st.sidebar.number_input("Strike Zone Y2", value=350)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This tool uses the local YOLOv8s model for better accuracy. "
        "Adjust the confidence threshold if the ball is not being detected."
    )

    uploaded_file = st.file_uploader("Upload a short video of a pitch (MP4 format)", type="mp4")

    if uploaded_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyze Pitch"):
            # Pass the confidence threshold to the analysis function
            output_video_path, annotator = analyze_video(temp_video_path, strike_zone_coords, confidence_threshold)
            if output_video_path:
                st.success("Analysis complete!")
                st.video(output_video_path)

                # --- 2D Strike Zone Visualization ---
                strike_zone_vis = np.zeros((500, 600, 3), dtype="uint8")
                cv2.rectangle(strike_zone_vis, (strike_zone_coords['x1'], strike_zone_coords['y1']), (strike_zone_coords['x2'], strike_zone_coords['y2']), (255, 0, 0), 2)

                trajectory_points = [pts[0] for pts in annotator.buffer if pts.size > 0]
                
                graph_caption = "2D Strike Zone (No trajectory detected)"
                if len(trajectory_points) > 1:
                    graph_caption = "2D Strike Zone with Pitch Trajectory"
                    pts_np = np.array(trajectory_points, dtype=np.int32)
                    cv2.polylines(strike_zone_vis, [pts_np], isClosed=False, color=(0, 0, 255), thickness=2)
                    for point in pts_np:
                        cv2.circle(strike_zone_vis, tuple(point), 3, (0, 255, 255), -1)

                st.image(strike_zone_vis, caption=graph_caption, use_container_width=True)

                if trajectory_points:
                    last_point = trajectory_points[-1]
                    st.write(f"Estimated Final Position: ({last_point[0]}, {last_point[1]})")
                else:
                    st.write("No final position could be estimated as no ball was tracked.")

if __name__ == "__main__":
    main()
