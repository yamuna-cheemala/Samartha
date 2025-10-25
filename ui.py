import streamlit as st
import cv2
import numpy as np
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Yoga Pose Correction",
    page_icon="🧘 Yoga Pose Correction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        width: 100%;
        height: 80px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .status-perfect {
        background-color: #28a745;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .status-adjust {
        background-color: #dc3545;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .instructions {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Configuration Constants ---
CORRECTNESS_TOLERANCE = 0.15
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
        st.error(f"Error: {json_path} not found.")
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

# --- Pose Checking ---
def check_pose_correctness(user_kps, ref_kps, tolerance=CORRECTNESS_TOLERANCE):
    incorrect_edges = []
    user_kps_array = np.array(user_kps)
    ref_kps_array = np.array(ref_kps)
    for a, b in EDGES:
        dist_a = np.linalg.norm(user_kps_array[a] - ref_kps_array[a])
        dist_b = np.linalg.norm(user_kps_array[b] - ref_kps_array[b])
        if dist_a > tolerance or dist_b > tolerance:
            incorrect_edges.append((a, b))
    return incorrect_edges

# --- Draw Skeleton ---
def draw_user_skeleton(frame, user_kps, incorrect_edges, ref_kps):
    h, w, _ = frame.shape
    # Draw reference pose (yellow)
    for y, x in ref_kps:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)
    for a, b in EDGES:
        y1, x1 = ref_kps[a]
        y2, x2 = ref_kps[b]
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 255), 2)
    # Draw user skeleton
    for y, x in user_kps:
        cv2.circle(frame, (int(x * w), int(y * h)), 6, (255, 255, 255), -1)
    for a, b in EDGES:
        y1, x1 = user_kps[a]
        y2, x2 = user_kps[b]
        color = (0, 0, 255) if (a, b) in incorrect_edges or (b, a) in incorrect_edges else (0, 255, 0)
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 4)
    return frame

# --- Pose Detection Transformer ---
class PoseTransformer(VideoTransformerBase):
    def __init__(self, model, ref_kps):
        self.model = model
        self.ref_kps = ref_kps

    def transform(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tf = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        img_tf = tf.cast(img_tf, dtype=tf.int32)
        outputs = self.model.signatures['serving_default'](img_tf)
        user_kps = outputs['output_0'].numpy()[0, 0, :, :2]
        # Ground reference to user's floor
        y_shift = np.max(user_kps[:,0]) - np.max(self.ref_kps[:,0])
        grounded_ref_kps = self.ref_kps.copy()
        grounded_ref_kps[:,0] += y_shift
        incorrect_edges = check_pose_correctness(user_kps, grounded_ref_kps)
        frame = draw_user_skeleton(frame, user_kps, incorrect_edges, grounded_ref_kps)
        # Add status text
        is_correct = len(incorrect_edges) == 0
        status_text = "PERFECT! ✓" if is_correct else "ADJUST POSE"
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return frame

# --- Main App ---
def main():
    st.markdown("<h1 style='text-align: center; color: white;'>🧘 Yoga Pose Correction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Select a pose and match it with real-time feedback</p>", unsafe_allow_html=True)

    reference_poses = load_reference_poses()
    if not reference_poses:
        st.error("No reference poses found.")
        return

    model = load_movenet_model()

    pose_names = list(reference_poses.keys())

    # Instructions
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        <div class="instructions">
        <ol>
            <li>Select a pose</li>
            <li>Allow camera access</li>
            <li>Match your pose to the yellow skeleton</li>
            <li>Green lines = correct, Red lines = adjust</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    selected_pose = st.selectbox("Select Pose", pose_names)

    if st.button("Start Camera"):
        ref_kps = reference_poses[selected_pose]
        webrtc_streamer(key="yoga", video_transformer_factory=lambda: PoseTransformer(model, ref_kps))

if __name__ == "__main__":
    main()
