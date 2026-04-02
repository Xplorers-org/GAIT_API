from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, detrend
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import base64
import io

app = FastAPI(
    title="HEMAS NeuroTrack Gait Analysis API",
    description="Clinical gait analysis using MediaPipe Pose estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Create output directory(1)
# OUTPUT_DIR = Path("/tmp/gait_outputs")
# OUTPUT_DIR.mkdir(exist_ok=True)

# Create output directory(2) (cross-platform, inside repository)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "runs" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def smooth_signal(data, window_length=9, polyorder=3):
    """Applies Savitzky-Golay filter to remove MediaPipe tracking jitter."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)


def extract_validate_and_visualize(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    signals = {
        'l_ankle_y': [], 'r_ankle_y': [],
        'l_arm_swing': [], 'r_arm_swing': [],
        'mid_hip_x': [], 'mid_hip_y': [],
        'l_foot_x': [], 'r_foot_x': []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # BODY CENTER
            mid_hip_x = (lm[23].x + lm[24].x) / 2
            mid_hip_y = (lm[23].y + lm[24].y) / 2

            signals['mid_hip_x'].append(mid_hip_x)
            signals['mid_hip_y'].append(mid_hip_y)

            # LOWER BODY
            signals['l_ankle_y'].append(lm[27].y)
            signals['r_ankle_y'].append(lm[28].y)

            # Normalize foot X relative to body center
            signals['l_foot_x'].append(lm[31].x - mid_hip_x)
            signals['r_foot_x'].append(lm[32].x - mid_hip_x)

            # ARM SWING
            l_torso_len = np.linalg.norm([
                lm[11].x - lm[23].x,
                lm[11].y - lm[23].y
            ])
            r_torso_len = np.linalg.norm([
                lm[12].x - lm[24].x,
                lm[12].y - lm[24].y
            ])

            # Relative to shoulder
            l_ws = np.linalg.norm([
                lm[15].x - lm[11].x,
                lm[15].y - lm[11].y
            ])
            r_ws = np.linalg.norm([
                lm[16].x - lm[12].x,
                lm[16].y - lm[12].y
            ])

            # Normalized + stabilized
            signals['l_arm_swing'].append(l_ws / (l_torso_len + 1e-6))
            signals['r_arm_swing'].append(r_ws / (r_torso_len + 1e-6))

            # DRAW SKELETON
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255), thickness=4, circle_radius=4
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2
                )
            )

        out.write(image_bgr)

    cap.release()
    out.release()

    # VALIDATION
    if len(signals['mid_hip_x']) == 0:
        raise ValueError("❌ No person detected in the video.")

    # IMPROVED SIDE VIEW DETECTION
    x_var = np.var(signals['mid_hip_x'])
    y_var = np.var(signals['mid_hip_y'])

    if x_var > y_var:
        raise ValueError("❌ SIDE-VIEW DETECTED: Upload FRONT-VIEW video")

    # SMOOTH SIGNALS
    for key in signals:
        signals[key] = smooth_signal(np.array(signals[key]))

    return signals, fps


def robust_amplitude(signal, threshold=0.01):
    """Computes real movement amplitude and removes MediaPipe noise."""
    if len(signal) == 0:
        return 0
    amp = np.percentile(signal, 95) - np.percentile(signal, 5)
    return amp if amp > threshold else 0


def compute_gait_features(signals, fps):
    features = {}

    # FOOT X SIGNAL
    l_signal = detrend(signals['l_foot_x'])
    r_signal = detrend(signals['r_foot_x'])

    def smooth(x):
        return np.convolve(x, np.ones(7)/7, mode='same')

    l_signal = smooth(l_signal)
    r_signal = smooth(r_signal)

    # PEAK DETECTION
    min_distance = int(fps * 0.3)

    l_peaks, _ = find_peaks(
        l_signal,
        distance=min_distance,
        prominence=np.std(l_signal) * 0.25
    )

    r_peaks, _ = find_peaks(
        r_signal,
        distance=min_distance,
        prominence=np.std(r_signal) * 0.25
    )

    # CLEAN PEAKS
    def clean_peaks(peaks, fps, min_gap=0.4):
        if len(peaks) == 0:
            return peaks

        cleaned = [peaks[0]]
        for p in peaks[1:]:
            if (p - cleaned[-1]) / fps > min_gap:
                cleaned.append(p)
        return np.array(cleaned)

    l_peaks = clean_peaks(l_peaks, fps)
    r_peaks = clean_peaks(r_peaks, fps)

    # STRIDE TIMES
    l_stride = np.diff(l_peaks) / fps if len(l_peaks) > 1 else np.array([])
    r_stride = np.diff(r_peaks) / fps if len(r_peaks) > 1 else np.array([])

    # ROBUST FILTER
    def filter_stride(strides):
        if len(strides) < 2:
            return strides

        median = np.median(strides)

        filtered = strides[
            (strides > 0.4) & (strides < 1.3) &
            (np.abs(strides - median) < 0.15)
        ]

        return filtered

    l_stride = filter_stride(l_stride)
    r_stride = filter_stride(r_stride)

    # STRIDE VARIABILITY
    stride_variability = None

    if len(l_stride) >= 2 and len(r_stride) >= 2:
        cv_left = np.std(l_stride) / np.median(l_stride)
        cv_right = np.std(r_stride) / np.median(r_stride)
        stride_variability = ((cv_left + cv_right) / 2) * 100
    elif len(l_stride) >= 2:
        stride_variability = (np.std(l_stride) / np.median(l_stride)) * 100
    elif len(r_stride) >= 2:
        stride_variability = (np.std(r_stride) / np.median(r_stride)) * 100
    else:
        stride_variability = 0.5

    stride_variability = max(0.5, min(stride_variability, 8.5))
    features['stride_variability'] = stride_variability

    # CADENCE
    total_steps = len(l_peaks) + len(r_peaks)
    duration_minutes = len(l_signal) / fps / 60
    cadence = total_steps / duration_minutes if duration_minutes > 0 else 0
    features['cadence'] = cadence

    # SYMMETRY
    if len(l_stride) > 0 and len(r_stride) > 0:
        l_mean = np.mean(l_stride)
        r_mean = np.mean(r_stride)
        symmetry = abs(l_mean - r_mean) / ((l_mean + r_mean) / 2)
    else:
        symmetry = 0

    features['symmetry_ratio'] = symmetry

    # ARM SWING
    l_arm = smooth(signals['l_arm_swing'])
    r_arm = smooth(signals['r_arm_swing'])

    l_amp = robust_amplitude(l_arm)
    r_amp = robust_amplitude(r_arm)

    scale_factor = 20.0
    l_amp *= scale_factor
    r_amp *= scale_factor

    avg_arm = (l_amp + r_amp) / 2

    features['l_arm_amp'] = l_amp
    features['r_arm_amp'] = r_amp
    features['avg_arm_swing'] = avg_arm

    # ARM ASYMMETRY
    if l_amp > 0 and r_amp > 0:
        asym = abs(l_amp - r_amp) / max(l_amp, r_amp) * 100
    else:
        asym = 100

    features['arm_asymmetry_index'] = asym

    # Save signals for plots
    signals['l_signal'] = l_signal
    signals['r_signal'] = r_signal

    return features, l_peaks, r_peaks


def interpret_clinical_features(features, gender):
    """Generate clinical interpretation text"""
    interpretation = []
    
    interpretation.append("=" * 50)
    interpretation.append(f" HEMAS NEUROTRACK: CLINICAL INTERPRETATION ({gender.upper()})")
    interpretation.append("=" * 50)

    # Stride Variability
    cv = features['stride_variability']
    interpretation.append(f"\n▶ STRIDE TIME VARIABILITY: {cv:.2f}%")
    if gender.lower() == 'male':
        if cv <= 2.5:
            interpretation.append("   ↳ Status: NORMAL (Healthy rhythm)")
        elif cv <= 4.0:
            interpretation.append("   ↳ Status: MILD DEVIATION (Slight irregularity)")
        elif cv <= 6.0:
            interpretation.append("   ↳ Status: MODERATE IMPAIRMENT (Noticeable rhythm fluctuation)")
        else:
            interpretation.append("   ↳ Status: HIGH IMPAIRMENT (Severe gait instability detected)")
    elif gender.lower() == 'female':
        if cv <= 3.0:
            interpretation.append("   ↳ Status: NORMAL (Healthy rhythm)")
        elif cv <= 4.5:
            interpretation.append("   ↳ Status: MILD DEVIATION (Slight irregularity)")
        elif cv <= 6.5:
            interpretation.append("   ↳ Status: MODERATE IMPAIRMENT (Noticeable rhythm fluctuation)")
        else:
            interpretation.append("   ↳ Status: HIGH IMPAIRMENT (Severe gait instability detected)")

    # Cadence
    cad = features['cadence']
    interpretation.append(f"\n▶ CADENCE: {cad:.1f} steps/min")
    if gender.lower() == 'male':
        if cad >= 100:
            interpretation.append("   ↳ Status: NORMAL (Healthy pace)")
        elif cad >= 90:
            interpretation.append("   ↳ Status: MILD REDUCTION (Slightly slower pace)")
        elif cad >= 80:
            interpretation.append("   ↳ Status: MODERATE REDUCTION (Bradykinesia indicator)")
        else:
            interpretation.append("   ↳ Status: HIGH REDUCTION (Severe shuffling or freezing tendency)")
    elif gender.lower() == 'female':
        if cad >= 105:
            interpretation.append("   ↳ Status: NORMAL (Healthy pace)")
        elif cad >= 95:
            interpretation.append("   ↳ Status: MILD REDUCTION (Slightly slower pace)")
        elif cad >= 85:
            interpretation.append("   ↳ Status: MODERATE REDUCTION (Bradykinesia indicator)")
        else:
            interpretation.append("   ↳ Status: HIGH REDUCTION (Severe shuffling or freezing tendency)")

    # Symmetry
    interpretation.append("\n▶ GAIT SYMMETRY:")
    sym = features['symmetry_ratio']
    if sym >= 0.95:
        interpretation.append("   ↳ Status: HIGHLY SYMMETRIC (Healthy left/right balance)")
    elif sym >= 0.85:
        interpretation.append("   ↳ Status: MILD ASYMMETRY (Slight favoring of one leg)")
    else:
        interpretation.append("   ↳ Status: SIGNIFICANT ASYMMETRY (Typical of unilateral Parkinsonian symptoms)")

    # Arm Swing
    interpretation.append("\n▶ OVERALL ARM SWING:")
    swing = features['avg_arm_swing']
    interpretation.append(f"   [Raw AI Swing Variance Score: {swing:.2f}]")

    if swing > 5.0:
        interpretation.append("   ↳ Status: HEALTHY RANGE OF MOTION (Fluid arm swing)")
    elif swing > 2.5:
        interpretation.append("   ↳ Status: REDUCED AMPLITUDE (Stiffened arm movement)")
    else:
        interpretation.append("   ↳ Status: SEVERELY RESTRICTED (En-bloc / Rigid posture detected)")

    # Arm Asymmetry
    interpretation.append("\n▶ ARM SWING ASYMMETRY:")
    arm_asym = features['arm_asymmetry_index']
    interpretation.append(f"   [Raw AI Asymmetry Index: {arm_asym:.1f}%]")

    if arm_asym <= 25.0:
        interpretation.append("   ↳ Status: BALANCED (Both arms swing/rest equally)")
    elif arm_asym <= 45.0:
        interpretation.append("   ↳ Status: MILD ASYMMETRY (One arm shows slight rigidity)")
    else:
        interpretation.append("   ↳ Status: UNILATERAL RIGIDITY (One arm is significantly stiffer than the other)")

    interpretation.append("\n" + "=" * 50)
    
    return "\n".join(interpretation)


def score_stride_variability(v):
    if v <= 2:
        return 100
    elif v <= 4:
        return 80
    elif v <= 6:
        return 60
    elif v <= 8.5:
        return 40
    else:
        return 20


def score_symmetry(s):
    if s < 0.05:
        return 100
    elif s < 0.1:
        return 80
    elif s < 0.2:
        return 60
    elif s < 0.3:
        return 40
    else:
        return 20


def score_cadence(c):
    if 100 <= c <= 115:
        return 100
    elif 90 <= c < 100 or 115 < c <= 125:
        return 80
    elif 80 <= c < 90 or 125 < c <= 135:
        return 60
    else:
        return 40


def score_arm_swing(a):
    if a > 1.5:
        return 100
    elif a > 1.0:
        return 80
    elif a > 0.5:
        return 60
    elif a > 0.2:
        return 40
    else:
        return 20


def score_arm_asymmetry(a):
    if a < 10:
        return 100
    elif a < 20:
        return 80
    elif a < 40:
        return 60
    elif a < 60:
        return 40
    else:
        return 20


def compute_gait_stability_score(features):
    sv = features['stride_variability']
    sym = features['symmetry_ratio']
    cad = features['cadence']
    arm = features['avg_arm_swing']
    asym = features['arm_asymmetry_index']

    sv_score = score_stride_variability(sv)
    sym_score = score_symmetry(sym)
    cad_score = score_cadence(cad)
    arm_score = score_arm_swing(arm)
    asym_score = score_arm_asymmetry(asym)

    final_score = (
        0.30 * sv_score +
        0.20 * sym_score +
        0.15 * cad_score +
        0.20 * arm_score +
        0.15 * asym_score
    )

    return round(final_score, 2)


def interpret_gait_score(score):
    if score >= 85:
        return "🟢 Normal gait (Stable)"
    elif score >= 70:
        return "🟡 Mild impairment"
    elif score >= 55:
        return "🟠 Moderate impairment"
    else:
        return "🔴 Severe gait instability"


def plot_clinical_biomarkers(signals, features, l_peaks, r_peaks, fps, output_path):
    """Generate clinical visualization dashboard and save to file"""
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('NeuroTrack AI: Kinematic Gait Analysis', fontsize=20, fontweight='bold', color='#1f77b4')

    time_axis = np.arange(len(signals['l_ankle_y'])) / fps

    # 1. Ankle Vertical Displacement
    axs[0, 0].plot(time_axis, signals['l_ankle_y'], label='Left Ankle', color='blue', alpha=0.7)
    axs[0, 0].plot(time_axis, signals['r_ankle_y'], label='Right Ankle', color='orange', alpha=0.7)
    axs[0, 0].set_title('Ankle Vertical Displacement')
    axs[0, 0].invert_yaxis()
    axs[0, 0].legend()

    # 2. Peak Detection
    if 'l_signal' in signals and 'r_signal' in signals:
        axs[0, 1].plot(time_axis, signals['l_signal'], color='gray', alpha=0.6)

        if len(l_peaks) > 0:
            axs[0, 1].plot(time_axis[l_peaks], signals['l_signal'][l_peaks], "X",
                           color='red', markersize=8, label='Left Steps')

        if len(r_peaks) > 0:
            axs[0, 1].plot(time_axis[r_peaks], signals['r_signal'][r_peaks], "X",
                           color='green', markersize=8, label='Right Steps')

        axs[0, 1].set_title('Step Detection (Foot X Signal)')
        axs[0, 1].legend()
    else:
        axs[0, 1].set_title("Step Detection (No Data)")

    # 3. Stride Times
    l_stride_times = np.diff(l_peaks) / fps if len(l_peaks) > 1 else []
    r_stride_times = np.diff(r_peaks) / fps if len(r_peaks) > 1 else []

    if len(l_stride_times) > 0:
        axs[1, 0].plot(l_stride_times, marker='o', linestyle='-', color='blue', label='Left')

    if len(r_stride_times) > 0:
        axs[1, 0].plot(r_stride_times, marker='o', linestyle='-', color='orange', label='Right')

    axs[1, 0].set_title(f"Stride Variability (CV: {features['stride_variability']:.2f}%)")
    axs[1, 0].legend()

    # 4. Arm Swing
    axs[1, 1].plot(time_axis, signals['l_arm_swing'], label='Left Arm', color='purple', alpha=0.7)
    axs[1, 1].plot(time_axis, signals['r_arm_swing'], label='Right Arm', color='brown', alpha=0.7)
    axs[1, 1].set_title('Normalized Arm Swing')
    axs[1, 1].legend()

    # 5. Arm Amplitude
    axs[2, 0].bar(
        ['Left Arm', 'Right Arm'],
        [features['l_arm_amp'], features['r_arm_amp']],
        color=['purple', 'brown']
    )
    axs[2, 0].set_title(f"Arm Asymmetry Index: {features['arm_asymmetry_index']:.1f}%")
    axs[2, 0].set_ylabel('Amplitude')

    # 6. Postural Sway
    axs[2, 1].plot(time_axis, signals['mid_hip_x'], color='teal')
    axs[2, 1].set_title('Postural Sway (Hip X Movement)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HEMAS NeuroTrack Gait Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_gait": "/analyze",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.post("/analyze")
async def analyze_gait(
    video: UploadFile = File(..., description="Video file for gait analysis"),
    gender: str = Form(..., description="Patient gender (male/female)")
):
    """
    Analyze gait from video file
    
    - **video**: Video file (mp4, mov, avi, etc.)
    - **gender**: Patient gender (male or female) for clinical interpretation
    
    Returns:
    - Annotated video with skeleton overlay
    - Clinical biomarkers visualization
    - Detailed clinical interpretation
    - Gait stability score
    """
    
    if gender.lower() not in ['male', 'female']:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    # Create temporary file paths without keeping open handles (important on Windows)
    input_fd, temp_input_path = tempfile.mkstemp(suffix='.mp4')
    output_fd, temp_output_video_path = tempfile.mkstemp(suffix='.mp4')
    plot_fd, temp_plot_path = tempfile.mkstemp(suffix='.png')
    os.close(input_fd)
    os.close(output_fd)
    os.close(plot_fd)
    
    try:
        # Save uploaded video
        content = await video.read()
        with open(temp_input_path, 'wb') as f:
            f.write(content)
        
        # Process video
        print("1. Overlaying skeleton and extracting kinematics...")
        signals, fps = extract_validate_and_visualize(temp_input_path, temp_output_video_path)
        
        print("2. Computing clinical biomarkers...")
        features, l_peaks, r_peaks = compute_gait_features(signals, fps)
        
        # Generate clinical interpretation
        clinical_interpretation = interpret_clinical_features(features, gender)
        
        # Compute gait stability score
        score = compute_gait_stability_score(features)
        interpretation = interpret_gait_score(score)
        
        features['gait_score'] = score
        features['gait_interpretation'] = interpretation
        
        # Generate plots
        print("3. Generating Clinical Visualization Dashboard...")
        plot_clinical_biomarkers(signals, features, l_peaks, r_peaks, fps, temp_plot_path)
        
        # Convert files to base64
        with open(temp_output_video_path, 'rb') as f:
            annotated_video_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        with open(temp_plot_path, 'rb') as f:
            plot_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare response
        response = {
            "status": "success",
            "clinical_interpretation": clinical_interpretation,
            "gait_stability_score": score,
            "gait_interpretation": interpretation,
            "features": {
                "stride_variability": float(features['stride_variability']),
                "cadence": float(features['cadence']),
                "symmetry_ratio": float(features['symmetry_ratio']),
                "avg_arm_swing": float(features['avg_arm_swing']),
                "l_arm_amp": float(features['l_arm_amp']),
                "r_arm_amp": float(features['r_arm_amp']),
                "arm_asymmetry_index": float(features['arm_asymmetry_index'])
            },
            "files": {
                "annotated_video": f"data:video/mp4;base64,{annotated_video_b64}",
                "clinical_dashboard": f"data:image/png;base64,{plot_b64}"
            },
            "metadata": {
                "fps": float(fps),
                "total_frames": len(signals['l_ankle_y']),
                "duration_seconds": len(signals['l_ankle_y']) / fps,
                "left_steps_detected": int(len(l_peaks)),
                "right_steps_detected": int(len(r_peaks))
            }
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Cleanup
        for temp_file in [temp_input_path, temp_output_video_path, temp_plot_path]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except PermissionError:
                    pass


@app.post("/analyze_files")
async def analyze_gait_files(
    video: UploadFile = File(..., description="Video file for gait analysis"),
    gender: str = Form(..., description="Patient gender (male/female)")
):
    """
    Analyze gait from video file and return downloadable files
    
    - **video**: Video file (mp4, mov, avi, etc.)
    - **gender**: Patient gender (male or female) for clinical interpretation
    
    Returns:
    - JSON with URLs to download annotated video and clinical dashboard
    """
    
    if gender.lower() not in ['male', 'female']:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    # Create unique filenames
    import uuid
    session_id = str(uuid.uuid4())
    
    input_path = OUTPUT_DIR / f"{session_id}_input.mp4"
    output_video_path = OUTPUT_DIR / f"{session_id}_annotated.mp4"
    plot_path = OUTPUT_DIR / f"{session_id}_dashboard.png"
    
    try:
        # Save uploaded video
        content = await video.read()
        with open(input_path, 'wb') as f:
            f.write(content)
        
        # Process video
        signals, fps = extract_validate_and_visualize(str(input_path), str(output_video_path))
        features, l_peaks, r_peaks = compute_gait_features(signals, fps)
        
        # Generate clinical interpretation
        clinical_interpretation = interpret_clinical_features(features, gender)
        
        # Compute gait stability score
        score = compute_gait_stability_score(features)
        interpretation = interpret_gait_score(score)
        
        features['gait_score'] = score
        features['gait_interpretation'] = interpretation
        
        # Generate plots
        plot_clinical_biomarkers(signals, features, l_peaks, r_peaks, fps, str(plot_path))
        
        # Prepare response
        response = {
            "status": "success",
            "session_id": session_id,
            "clinical_interpretation": clinical_interpretation,
            "gait_stability_score": score,
            "gait_interpretation": interpretation,
            "features": {
                "stride_variability": float(features['stride_variability']),
                "cadence": float(features['cadence']),
                "symmetry_ratio": float(features['symmetry_ratio']),
                "avg_arm_swing": float(features['avg_arm_swing']),
                "l_arm_amp": float(features['l_arm_amp']),
                "r_arm_amp": float(features['r_arm_amp']),
                "arm_asymmetry_index": float(features['arm_asymmetry_index'])
            },
            "download_urls": {
                "annotated_video": f"/download/{session_id}_annotated.mp4",
                "clinical_dashboard": f"/download/{session_id}_dashboard.png"
            },
            "metadata": {
                "fps": float(fps),
                "total_frames": len(signals['l_ankle_y']),
                "duration_seconds": len(signals['l_ankle_y']) / fps,
                "left_steps_detected": int(len(l_peaks)),
                "right_steps_detected": int(len(r_peaks))
            }
        }
        
        # Clean up input file
        os.unlink(input_path)
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "HEMAS NeuroTrack API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
