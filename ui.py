import streamlit as st
import cv2
import numpy as np
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Yoga Pose Correction",
    page_icon="🧘 Yoga Pose Recog ition and Correction",
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
MIN_BODY_HEIGHT = 0.35
VISIBILITY_THRESHOLD = 0.15

# COCO keypoint edges
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

JOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# --- Initialize Session State ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reference_poses' not in st.session_state:
    st.session_state.reference_poses = {}
if 'selected_pose' not in st.session_state:
    st.session_state.selected_pose = None
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# --- Load Reference Poses ---
@st.cache_data
def load_reference_poses(json_path="keypoints_dataset.json"):
    """Loads and averages keypoints for each pose to create a stable reference."""
    if not os.path.exists(json_path):
        st.error(f"Error: {json_path} not found. Please ensure the dataset file exists.")
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

# --- Load Model ---
@st.cache_resource
def load_movenet_model():
    """Load MoveNet model from TensorFlow Hub"""
    try:
        with st.spinner("Loading MoveNet model... This may take a minute on first run."):
            model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Pose Check ---
def check_pose_correctness(user_kps, ref_kps, tolerance=CORRECTNESS_TOLERANCE):
    """Check which edges are incorrect based on tolerance"""
    incorrect_edges = []
    user_kps_array = np.array(user_kps)
    ref_kps_array = np.array(ref_kps)
    
    for a, b in EDGES:
        dist_a = np.linalg.norm(user_kps_array[a] - ref_kps_array[a])
        dist_b = np.linalg.norm(user_kps_array[b] - ref_kps_array[b])
        
        if dist_a > tolerance or dist_b > tolerance:
            incorrect_edges.append((a, b))
    
    return incorrect_edges

# --- MoveNet Detection ---
def detect_pose(frame, model):
    """Detect pose using MoveNet model"""
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    outputs = model.signatures['serving_default'](img)
    return outputs['output_0'].numpy()[0, 0, :, :2].tolist()

# --- Drawing Functions ---
def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    """Draw skeleton on frame"""
    h, w, _ = frame.shape
    for y, x in keypoints:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, color, -1)
    for a, b in EDGES:
        y1, x1 = keypoints[a]
        y2, x2 = keypoints[b]
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, thickness)

def draw_user_skeleton(frame, user_kps, incorrect_edges, ref_kps):
    """Draw user skeleton with color-coded feedback"""
    h, w, _ = frame.shape
    
    # Draw Yellow Reference Pose (Target)
    draw_skeleton(frame, ref_kps, color=(0, 255, 255), thickness=2)

    # Draw User Skeleton joints (White)
    for y, x in user_kps:
        cv2.circle(frame, (int(x * w), int(y * h)), 6, (255, 255, 255), -1)
    
    # Draw User Skeleton edges (Red for incorrect, Green for correct)
    for a, b in EDGES:
        y1, x1 = user_kps[a]
        y2, x2 = user_kps[b]
        
        # Color: Red for mistake, Green for correct
        color = (0, 0, 255) if (a, b) in incorrect_edges or (b, a) in incorrect_edges else (0, 255, 0)
        cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 4)
    
    return frame

# --- Callback Functions ---
def start_pose(pose_name):
    """Callback to start camera with selected pose"""
    st.session_state.selected_pose = pose_name
    st.session_state.run_camera = True

def stop_camera():
    """Callback to stop camera"""
    st.session_state.run_camera = False
    st.session_state.selected_pose = None

# --- Main App ---
def main():
    # Title
    st.markdown("<h1 style='text-align: center; color: white;'>🧘 Yoga Pose Correction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Select a pose and match it with real-time feedback</p>", unsafe_allow_html=True)
    
    # Load model and reference poses
    if st.session_state.model is None:
        st.session_state.model = load_movenet_model()
    
    if not st.session_state.reference_poses:
        st.session_state.reference_poses = load_reference_poses()
    
    if not st.session_state.reference_poses:
        st.error("No reference poses found. Please check your keypoints_dataset.json file.")
        return
    
    if st.session_state.model is None:
        st.error("Model failed to load. Please check your internet connection and try again.")
        return
    
    # Get list of poses
    pose_names = list(st.session_state.reference_poses.keys())
    
    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.markdown("""
        <div class="instructions">
        <h3>How to use:</h3>
        <ol>
            <li>Click on one of the pose buttons below</li>
            <li>Allow camera access when prompted</li>
            <li>Position yourself so your full body is visible</li>
            <li><strong>Yellow skeleton</strong> = Target pose to match</li>
            <li><strong>Green lines</strong> = Correct position ✓</li>
            <li><strong>Red lines</strong> = Need adjustment ✗</li>
            <li>Adjust your pose until all lines turn green!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show pose buttons only if camera is not running
    if not st.session_state.run_camera:
        st.markdown("<h2 style='text-align: center; color: white;'>Select Your Pose</h2>", unsafe_allow_html=True)
        
        # Create columns for pose buttons
        cols = st.columns(min(5, len(pose_names)))
        
        for idx, pose_name in enumerate(pose_names):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                st.button(
                    f"🧘 {pose_name.upper()}", 
                    key=f"pose_{pose_name}",
                    on_click=start_pose,
                    args=(pose_name,),
                    use_container_width=True
                )
        
        st.markdown("---")
    
    # If camera should run, show camera feed
    if st.session_state.run_camera and st.session_state.selected_pose:
        selected_pose = st.session_state.selected_pose
        ref_kps_target = st.session_state.reference_poses[selected_pose]
        
        st.markdown(f"<h2 style='text-align: center; color: white;'>Current Pose: {selected_pose.upper()}</h2>", unsafe_allow_html=True)
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Stop button at the top
            st.button("⏹️ Stop Camera", key="stop_camera", on_click=stop_camera, use_container_width=True)
            
            # Placeholders for status and metrics
            status_placeholder = st.empty()
            st.markdown("---")
            metrics_col1, metrics_col2 = st.columns(2)
            metric1_placeholder = metrics_col1.empty()
            metric2_placeholder = metrics_col2.empty()
        
        with col1:
            # Camera feed placeholder
            camera_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Please check your camera connection and permissions.")
            st.session_state.run_camera = False
            return
        
        # Process frames while camera should run
        frame_count = 0
        try:
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Convert to RGB for TensorFlow
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect pose
                user_kps_list = detect_pose(rgb_frame, st.session_state.model)
                user_kps_array = np.array(user_kps_list)
                
                # Ground reference pose to user's floor level
                lowest_user_y = np.max(user_kps_array[:, 0])
                lowest_ref_y = np.max(ref_kps_target[:, 0])
                y_shift = lowest_user_y - lowest_ref_y
                
                grounded_ref_kps = ref_kps_target.copy()
                grounded_ref_kps[:, 0] += y_shift
                grounded_ref_kps_list = grounded_ref_kps.tolist()
                
                # Check pose correctness
                incorrect_edges = check_pose_correctness(
                    user_kps_list, 
                    ref_kps_target, 
                    tolerance=CORRECTNESS_TOLERANCE
                )
                
                # Draw skeletons
                frame = draw_user_skeleton(frame, user_kps_list, incorrect_edges, grounded_ref_kps_list)
                
                # Add text overlay
                is_correct = len(incorrect_edges) == 0
                status_text = "PERFECT! ✓" if is_correct else "ADJUST POSE"
                text_color = (0, 255, 0) if is_correct else (0, 0, 255)
                
                cv2.putText(frame, f"Pose: {selected_pose.upper()}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Status: {status_text}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update status
                if is_correct:
                    status_placeholder.markdown(
                        "<div class='status-perfect'>🎉 PERFECT POSE! 🎉</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    status_placeholder.markdown(
                        "<div class='status-adjust'>⚠️ ADJUST YOUR POSE</div>", 
                        unsafe_allow_html=True
                    )
                
                # Update metrics
                accuracy = int((1 - len(incorrect_edges)/len(EDGES)) * 100)
                
                metric1_placeholder.markdown(f"""
                    <div class='metric-card'>
                        <h4>Incorrect Edges</h4>
                        <h2 style='color: {"#28a745" if is_correct else "#dc3545"};'>{len(incorrect_edges)}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                metric2_placeholder.markdown(f"""
                    <div class='metric-card'>
                        <h4>Accuracy</h4>
                        <h2 style='color: {"#28a745" if accuracy > 80 else "#ffc107" if accuracy > 60 else "#dc3545"};'>{accuracy}%</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Small delay
                time.sleep(0.03)  # ~30 FPS
                
                # Check if we should stop (button was pressed)
                if not st.session_state.run_camera:
                    break
                    
        except Exception as e:
            st.error(f"Error during pose detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()