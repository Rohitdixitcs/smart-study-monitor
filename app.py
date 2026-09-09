import streamlit as st
import cv2
import cvzone
import math
import base64
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO

# --- PREMIUM SMOOTH CSS ---
st.set_page_config(page_title="Smart Study Monitor", layout="wide", page_icon="📚")

st.markdown("""
<style>
    /* Smooth Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
        transition: background 0.5s ease;
    }
    
    /* Glowing Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        background: -webkit-linear-gradient(#00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 5px 0 0 0;
        font-size: 1.2rem;
        color: #d1d5db;
    }

    /* Status Cards with Smooth Transitions */
    .status-card {
        padding: 18px;
        border-radius: 15px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease-in-out;
    }
    
    /* Pulsing Glow */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 165, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0); }
    }

    .active-red { background: linear-gradient(45deg, #ff4b4b, #cc0000); color: white; animation: pulse-red 2s infinite; border: none; }
    .active-orange { background: linear-gradient(45deg, #ffa500, #cc8400); color: white; animation: pulse-orange 2s infinite; border: none; }
    .ok-green { background: linear-gradient(45deg, #28a745, #1e7e34); color: white; border: none; }
    .warn-yellow { background: linear-gradient(45deg, #ffc107, #d39e00); color: #212529; border: none; }

    /* Smooth Sidebar & Buttons */
    .stButton > button {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 12px;
        border-radius: 10px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: black;
        border: none;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    .stSidebar {
        background: rgba(0, 0, 0, 0.4);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Smooth Video Frame */
    div[data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
        border: 2px solid rgba(0, 255, 255, 0.3);
        transition: box-shadow 0.5s ease;
    }
    div[data-testid="stImage"]:hover {
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.4);
    }

    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 1rem;
        backdrop-filter: blur(5px);
        z-index: 1000;
    }
    .footer a {
        color: #00ffff;
        text-decoration: none;
        font-weight: bold;
        margin: 0 5px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>📚 Smart Study Monitor</h1>
    <p>Neural Focus System</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 🛠️ Control Center")
    start_camera = st.button("▶️ START MONITORING")
    stop_camera = st.button("⏹️ SHUT DOWN SYSTEM")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    sleep_threshold = st.slider("Eye Closure Sensitivity", 0.05, 0.5, 0.2, 0.01)
    phone_threshold = st.slider("Phone Detection Confidence", 0.0, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.info("💡 **Tip:** Click the Start button once to allow the browser to play looping alarms automatically.")

if stop_camera:
    st.stop()

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
    yolo_model = YOLO("yolov8s.pt") 
    return face_detector, hand_detector, yolo_model

if start_camera:
    with st.spinner("Initializing AI Neural Network..."):
        face_detector, hand_detector, model = load_models()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📹 Live Neural Feed")
        frame_placeholder = st.empty()

    with col2:
        st.markdown("### 🟢 Status")
        status_container = st.container()
        with status_container:
            status_sleep = st.empty()
            status_face = st.empty()
            status_phone = st.empty()
            status_general = st.empty()
            audio_placeholder = st.empty()

    # Added a safe check for camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access camera. Please close other apps using your camera and try again.")
        st.stop()
    
    cap.set(3, 640) 
    cap.set(4, 480) 
    
    current_playing = None

    def play_audio_loop(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f'<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

    while True:
        success, img = cap.read()
        if not success:
            st.error("Failed to access camera. Please check your permissions.")
            break

        img, faces = face_detector.findFaceMesh(img, draw=False)
        hands, img = hand_detector.findHands(img, draw=False)

        is_sleepy = False
        is_face_covered = False
        is_phone = False
        face_visible = False

        if faces:
            face_visible = True
            face = faces[0]
            def dist(p1, p2):
                return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            r_horiz = dist(face[33], face[133])
            r_vert = dist(face[160], face[144])
            right_ear = r_vert / r_horiz
            l_horiz = dist(face[362], face[263])
            l_vert = dist(face[385], face[380])
            left_ear = l_vert / l_horiz
            ear = (right_ear + left_ear) / 2
            if ear < sleep_threshold:
                is_sleepy = True

        if not face_visible and hands:
            is_face_covered = True

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 67 and conf > phone_threshold:
                    is_phone = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cvzone.putTextRect(img, "PHONE DETECTED", (x1, y1 - 10), scale=2, colorR=(0, 255, 255))

        # --- AUDIO LOGIC (UNTOUCHED) ---
        if is_sleepy:
            if current_playing != 'sleep':
                play_audio_loop("alarm.mp3")
                current_playing = 'sleep'
        elif is_face_covered:
            if current_playing != 'face':
                play_audio_loop("faudio.mp3")
                current_playing = 'face'
        elif is_phone:
            if current_playing != 'phone':
                play_audio_loop("paudio.mp3")
                current_playing = 'phone'
        else:
            if current_playing is not None:
                audio_placeholder.empty()
                current_playing = None

        frame_placeholder.image(img, channels="BGR", width='stretch')

        # --- STATUS UPDATES (UNTOUCHED) ---
        if is_sleepy:
            status_sleep.markdown('<div class="status-card active-red">😴 SLEEPING</div>', unsafe_allow_html=True)
        else:
            status_sleep.markdown('<div class="status-card ok-green">😊 Eyes Open</div>', unsafe_allow_html=True)

        if is_face_covered:
            status_face.markdown('<div class="status-card active-red">🙈 FACE COVERED</div>', unsafe_allow_html=True)
        elif not face_visible:
            status_face.markdown('<div class="status-card warn-yellow">👀 Searching...</div>', unsafe_allow_html=True)
        else:
            status_face.markdown('<div class="status-card ok-green">🙂 Face Visible</div>', unsafe_allow_html=True)

        if is_phone:
            status_phone.markdown('<div class="status-card active-orange">📱 PHONE DETECTED</div>', unsafe_allow_html=True)
        else:
            status_phone.markdown('<div class="status-card ok-green">📵 No Phone</div>', unsafe_allow_html=True)

        status_general.markdown("### ⚡ System Running...")

    cap.release()
    cv2.destroyAllWindows()
else:
    st.markdown("### 👈 Press **START MONITORING** in the sidebar to engage the AI.")

# --- ALWAYS VISIBLE FOOTER (YOUR CREDENTIALS) ---
st.markdown("""
<div class="footer">
    Made with ❤️ by <strong>Rohit Dixit</strong> | 
    <a href="https://www.linkedin.com/in/rohitdixitcs/" target="_blank">LinkedIn</a> | 
    <a href="https://rohitdixitdev.vercel.app/" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)