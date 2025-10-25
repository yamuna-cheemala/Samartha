import streamlit as st
import cv2
import numpy as np
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import av
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Yoga Pose Correction",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Configuration Constants ---
CORRECTNESS_TOLERANCE = 0.15

# COCO keypoint edges
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- Load Reference Poses ---
@st.cache_data
def load_reference_poses(json_path="keypoints_dataset.json"):
    if not os.path.exists(json_path):
        st.error(f"{json_path} not found")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    pose_data = {}
    for item in data.get("TRAIN", []):
        pose_name = item["pose"]
        keypoints = np.array(item["keypoints"])
        pose_data.setdefault(pose_name, []).append(keypoints)
    reference_poses = {pose: np.mean(kps_list, axis=0) for pose, kps_list in pose_data.items()}
    return reference_poses

# --- Load MoveNet Model ---
@st.cache_resource
def load_movenet_model():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model

# --- Pose Detection ---
def detect_pose(frame, model):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    outputs = model.signatures['serving_default'](img)
    return outputs['output_0'].numpy()[0, 0, :, :2].tolist()

# --- Skeleton Drawing ---
def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    h, w, _ = frame.shape
    for y, x in keypoints:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1)
    for a, b in EDGES:
        y1, x1 = keypoints[a]
        y2, x2 = keypoints[b]
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, thickness)

def check_pose_correctness(user_kps, ref_kps, tolerance=CORRECTNESS_TOLERANCE):
    incorrect_edges = []
    user_kps = np.array(user_kps)
    ref_kps = np.array(ref_kps)
    for a, b in EDGES:
        if np.linalg.norm(user_kps[a] - ref_kps[a]) > tolerance or np.linalg.norm(user_kps[b] - ref_kps[b]) > tolerance:
            incorrect_edges.append((a, b))
    return incorrect_edges

def draw_user_skeleton(frame, user_kps, incorrect_edges, ref_kps):
    draw_skeleton(frame, ref_kps, color=(0, 255, 255), thickness=2)  # reference pose
    h, w, _ = frame.shape
    for y, x in user_kps:
        cv2.circle(frame, (int(x * w), int(y * h)), 6, (255, 255, 255), -1)
    for a, b in EDGES:
        y1, x1 = user_kps[a]
        y2, x2 = user_kps[b]
        color = (0, 0, 255) if (a, b) in incorrect_edges else (0, 255, 0)
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 4)
    return frame

# --- WebRTC Video Transformer ---
class PoseTransformer(VideoTransformerBase):
    def __init__(self, model, ref_pose):
        self.model = model
        self.ref_pose = ref_pose

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        user_kps = detect_pose(rgb_img, self.model)
        incorrect_edges = check_pose_correctness(user_kps, self.ref_pose)
        out_frame = draw_user_skeleton(img, user_kps, incorrect_edges, self.ref_pose)
        return out_frame

# --- Main App ---
def main():
    st.title("🧘 Yoga Pose Correction (Live Camera)")

    # Load model & reference
    model = load_movenet_model()
    reference_poses = load_reference_poses()
    if not reference_poses:
        return
    pose_names = list(reference_poses.keys())

    selected_pose = st.selectbox("Select a Pose:", pose_names)
    if selected_pose:
        st.markdown("Allow camera access and maintain full body in view")
        webrtc_streamer(
            key="yoga-stream",
            video_transformer_factory=lambda: PoseTransformer(model, reference_poses[selected_pose]),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )

if __name__ == "__main__":
    main()
