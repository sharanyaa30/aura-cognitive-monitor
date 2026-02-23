# Cognitive Load Monitor — Full Project Summary

## Executive Overview

The **Cognitive Load Monitor** is a real-time desktop application that uses webcam-based biometric signals to measure and visualize a user's cognitive stress levels while working. By tracking eye blink patterns, head posture, and breathing dynamics, the system quantifies mental fatigue on a 0–100 scale and provides immediate interventions (alerts, actionable tips, AI-generated rescue plans) to prevent burnout and optimize productivity.

---

## 1. Problem & Motivation

**The Challenge:**
- Knowledge workers often experience undetected cognitive overload, leading to burnout, poor decision-making, and health issues
- Traditional productivity tools focus on *output* (time tracking, task completion) but ignore the *mental cost* of work
- There's a gap between feeling "busy" and understanding actual cognitive stress

**The Solution:**
Build a non-invasive, **real-time cognitive load detection system** that:
1. Captures biometric signals from the webcam (blink, posture, breathing)
2. Fuses these signals into a single, interpretable "load score"
3. Displays trends and thresholds visually
4. Triggers contextual interventions (tips, alerts, AI-generated action plans)

---

## 2. Technical Architecture

### 2.1 System Flow

```
Webcam Frame
    ↓
[CV Pipeline: Face Detection & Metrics]
    ├─ Blink Rate (Eye Aspect Ratio via MediaPipe)
    ├─ Head Posture (Nose landmark Z-depth)
    └─ Frame → Display
    ↓
[Biometric Input: Breathing Rate]
    └─ Rate simulation (future: Pose-based chest motion)
    ↓
[Load Detection: Rule-Based Scoring]
    ├─ Blink Component (0–50)
    ├─ Posture Penalty (0 or 20)
    ├─ Breathing Component (0–30)
    └─ Total Load Score (0–100)
    ↓
[Regulation & Intervention]
    ├─ Alert pyautogui (zoom, alerts)
    ├─ Trigger OpenAI summary → Rescue plan
    └─ Log to Event stream
    ↓
[Streamlit Dashboard]
    ├─ Live KPI display
    ├─ Historical trends (sparklines)
    ├─ Session analytics (pie chart, averages)
    ├─ Real-time alert banner
    ├─ Toast notifications
    └─ Event log
```

### 2.2 Load Score Formula (Detailed)

The cognitive load score is **purely rule-based** (no machine learning), computed by summing three independent heuristic components:

#### **Blink Component (0–50 points)**
- **Input:** Blink rate (blinks per minute)
- **Normal range:** 10–40 blings/min
- **Logic:**
  - ≤ 10 blinks/min → 0 points (very focused)
  - ≥ 40 blinks/min → 50 points (exhausted, eye strain)
  - Linear interpolation between
- **Formula:** $\text{blink\_score} = \min(1, \max(0, \frac{\text{blink\_rate} - 10}{30})) \times 50$
- **Why:** High blink rates indicate eye strain and mental fatigue

#### **Head Posture Component (0 or 20 points)**
- **Input:** Head forward flag (boolean)
- **Logic:**
  - Leaning forward detected → +20 points
  - Good posture → 0 points
- **Why:** Forward lean is a physical marker of stress and cognitive load (poor ergonomics → increased load)

#### **Breathing Component (0–30 points)**
- **Input:** Breathing rate (breaths per minute)
- **Normal range:** 12–20 bpm
- **Logic:**
  - Inside [12, 20] → 0 points
  - Outside range: penalty grows with deviation
  - Max penalty at ±8 bpm from boundary → 30 points
- **Formula:** $\text{breathing\_score} = \min(1, \frac{\text{deviation}}{8}) \times 30$
- **Why:** Abnormal breathing (shallow/rapid or deep/slow) correlates with stress response

#### **Total Score**
$$\text{load\_score} = \text{clamp}(\text{blink\_score} + \text{posture\_score} + \text{breathing\_score}, 0, 100)$$

#### **Demo Example**
- **User:** Working intensively on code
  - Blink rate: 30/min → 25 points
  - Head forward → 20 points
  - Breathing: 22 bpm (slightly elevated) → 15 points
  - **Total: 60 → "Normal" zone (yellow)**

---

## 3. Architecture & Module Breakdown

### 3.1 Computer Vision & Signal Processing Layer

**Files:** `src/cv/`, `src/signals/`

#### `src/cv/facial_features.py`
- **Core:** MediaPipe Face Mesh (468 landmarks)
- **Extracts:**
  - **Blink detection:** Eye Aspect Ratio (EAR) from 6 eye points per eye
    - EAR = $\frac{||p2-p6|| + ||p3-p5||}{2 \times ||p1-p4||}$
    - Closed threshold: EAR < 0.22, Open threshold: EAR > 0.25
    - Hysteresis to avoid jitter
    - Blink rate: count valid blinks in 60s window
  - **Head posture:** Nose tip Z-depth from landmark 1
    - Z < -0.08 → leaning forward
- **Return:** `{"blink_rate": float, "head_forward": bool}`

#### `src/cv/webcam_capture.py`
- Initialize OpenCV webcam stream
- Handle frame capture and release
- Camera permission checks

#### `src/cv/screen_ocr.py`
- Full-screen capture + Tesseract OCR
- Extract text context for AI summarization
- Used by regulation when load exceeds threshold

#### `src/signals/biometric_input.py`
- **Current:** Simulated breathing rate (sine wave, 10–25 bpm range)
- **Future:** MediaPipe Pose → chest/shoulder landmark motion analysis
- **Return:** `float` breathing rate in bpm

### 3.2 Detection & Regulation Layer

**Files:** `src/detection/`, `src/regulation/`

#### `src/detection/load_detector.py`
- Pure heuristic load score computation
- Three component scoring functions (as above)
- **Thresholds:**
  - Deep Flow: load < 35
  - Normal: 35 ≤ load < 70
  - Brain Fried: load ≥ 70

#### `src/regulation/workflow_regulator.py`
- **Triggered actions when load > 70:**
  1. Capture screen via OCR
  2. Send text to OpenAI API
  3. Get 3-bullet rescue plan ("Take a break", "Simplify task", etc.)
  4. Display modal alert with plan
  5. Log to event stream
- **Cooldown:** 10 seconds between actions (prevent spam)
- **Posture fix:** Ctrl++ zoom if head forward (reduce scale, force better posture)
- **API Integration:** OpenAI gpt-4.1-mini for context-aware summaries

#### `src/llm/openai_assistant.py`
- Prompt template: "Summarize what user is working on and give 3-bullet action plan"
- Called only when load exceeds 70
- Returns raw text response

### 3.3 Pipeline Orchestration

**File:** `src/pipeline.py`

- Synchronous real-time loop:
  1. Read webcam frame
  2. Extract face metrics (blink, posture)
  3. Request breathing rate signal
  4. Compute load score
  5. Return normalized dict: `{frame, blink_rate, head_forward, breathing_rate, load_score}`

---

## 4. Dashboard & User Interface

**File:** `app.py` (Streamlit)

### 4.1 Layout Structure

#### **Sidebar**
- Session duration timer
- Alert count
- Peak load tracker
- Threshold legend

#### **Header**
- Title, subtitle
- Real-time status badge (Deep Flow / Normal / Brain Fried with color)

#### **Row 1: KPI Cards (4 columns)**
- **Load Score** with color gradient
- **Blink Rate** (green < 15, yellow < 30, red ≥ 30)
- **Breathing Rate** (green 12–20, orange outside)
- **Posture** (green "Good" or red "Forward")
- Cards hover and animate

#### **Row 2: Live Webcam + Gauge**
- Left: Webcam feed (live)
- Right: Plotly gauge showing load score + contextual tips panel
  - Dynamic tips based on current state (e.g., posture, breathing, blink)

#### **Row 3: Real-Time Trends**
- 3 Plotly sparkline charts (updated every ~1 second to prevent flickering):
  - Load score trend (0–100)
  - Blink rate trend
  - Breathing rate trend
- SVG line fill with semi-transparency

#### **Row 4: Session Analytics**
- **Left:** Donut pie chart showing time distribution
  - Deep Flow (green)
  - Normal (yellow)
  - Brain Fried (red)
- **Right:** Smart recommendations log
  - AI-generated rescue plans (when load > 70)
  - Timestamped list

#### **Row 5: Event Log (Expandable)**
- All alerts, threshold crossings, and recommendations
- Reverse chronological order

#### **Alert Banner (Always Visible)**
- Real-time alerts for:
  - **Critical:** Load ≥ 70, Blink > 30
  - **Warning:** Load ≥ 35, Breathing out of range, Head forward
  - **OK:** All metrics healthy
- Toast notifications (pop-up) with 5-second cooldown
- Styled with color-coded left borders

### 4.2 State Management

**Session State Tracked:**
- `history_ts`, `history_blink`, `history_breathing`, `history_load`, `history_head` — 600-frame rolling history (≈10 min)
- `peak_load`, `peak_blink` — session peaks
- `time_deep_flow`, `time_normal`, `time_brain_fried` — zone time accounting
- `recommendations` — deque of timestamped AI plans + alerts
- `prev_alerts` — tracks which alerts were active to generate new ones only on state change

### 4.3 Performance Optimizations

- **Frame throttling:** Webcam + KPI updates every frame (~80 ms). Heavy Plotly charts update every ~12 frames (~1 s).
- **st.empty() containers:** Pre-allocated placeholders prevent DOM churn
- **Frame-count-based keys:** `key=f"g_{frame_count}"` ensures unique per-render keys without collision
- **Cooldown on toasts:** 5-second minimum between notifications prevents alert fatigue

---

## 5. Team Roles & Ownership

### **Person 1: Computer Vision & Signal Processing**
**Owns:** `src/cv/`, `src/signals/`
- MediaPipe Face Mesh integration
- Blink rate detection (EAR calculation, hysteresis)
- Head posture estimation (Z-depth)
- Screen OCR (Tesseract)
- Breathing simulation (future: Pose-based)
- **Skills:** OpenCV, image processing, biometric signal analysis

### **Person 2: Detection Logic, Regulation & AI**
**Owns:** `src/detection/`, `src/regulation/`, `src/llm/`, `src/pipeline.py`
- Load score heuristics (3-component formula)
- Threshold tuning and validation
- Workflow regulation (cooldowns, actions)
- OpenAI API integration
- Pipeline orchestration
- **Skills:** Algorithm design, heuristic tuning, API integration

### **Person 3: Dashboard, UI & Integration**
**Owns:** `app.py`, `src/config.py`, `requirements.txt`
- Streamlit dashboard design and layout
- Plotly chart integration
- Custom CSS styling
- Session state management
- Real-time metric visualization
- Alert/notification display
- Deployment & dependency management
- **Skills:** Frontend/UI design, Streamlit, Plotly, CSS

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI Framework** | Streamlit | Dashboard, real-time updates |
| **Charts** | Plotly | Interactive, responsive visualizations |
| **Webcam** | OpenCV (cv2) | Frame capture |
| **Face Detection** | MediaPipe Face Mesh | 468 landmark landmarks |
| **OCR** | Tesseract + Pillow | Screen text capture |
| **LLM** | OpenAI gpt-4.1-mini | Context-aware rescue plans |
| **Automation** | pyautogui | System-level interventions |
| **Data Structures** | Python `deque` | Efficient fixed-size rolling history |
| **Env Mgmt** | python-dotenv | API key management |

---

## 7. Data Flow Example: A 100-Second Session

**Time 0 s:** User starts app
- Frame rate: 12.5 fps (~80 ms per iteration)
- Session starts, empty history

**Time 10 s:** User is focused on coding
- Blink rate: 12/min (normal)
- Head posture: good (Z > -0.08)
- Breathing: 16 bpm (normal)
- Load score: 0 points → **Deep Flow** (green)
- KPI cards show green

**Time 45 s:** User struggling with complex algorithm
- Blink rate: 28/min (eye strain)
- Head posture: forward lean detected (Z < -0.08)
- Breathing: 24 bpm (elevated)
- Load score: 23 + 20 + 12 = **55** → **Normal** (yellow)
- KPI cards show yellow/orange
- Trend charts show rising load

**Time 70 s:** User hits cognitive wall
- Blink rate: 35/min (exhaustion)
- Head posture: severe forward lean
- Breathing: 28 bpm (stress breathing)
- Load score: 42 + 20 + 30 = **92 → clamped to 100** → **Brain Fried** (red)
- Alert banner: "CRITICAL: Cognitive load is 92. Take a break."
- Toast: "Cognitive load exceeded 70"
- `apply_regulation()` triggers:
  - Captures screen → "User is writing recursive function for graph..."
  - Calls OpenAI → Gets "1) Test with smaller input 2) Debug print statements 3) Step back 5 min"
  - Modal alert displayed
  - Event logged with timestamp

**Time 100 s:** User follows advice, takes break
- Blink rate drops to 18/min
- Head position improves
- Breathing normalizes to 15 bpm
- Load score: **22 → Deep Flow** (green)
- Alert banner: "All clear — all metrics within limits."

---

## 8. Key Features & Novelty

1. **Multi-Modal Biometric Fusion:** Combines 3 independent signals (blink, posture, breathing) into a single interpretable score
2. **Real-Time Dashboard:** Live visualization with sub-second latency
3. **Context-Aware AI Interventions:** OpenAI integration to provide personalized rescue plans based on screen content
4. **Non-Invasive:** Webcam-only, no wearables or hardware
5. **Rule-Based (No ML):** Transparent, tunable heuristics (easier to debug and explain than black-box models)
6. **Session Analytics:** Historical trends, zone distribution, peak tracking
7. **Dual-Layer Alerts:** Visual banner + toast notifications with smart cooldowns
8. **Modular Architecture:** Clean separation: CV → Detection → Regulation → UI

---

## 9. Future Enhancements

- **Real breathing detection** via MediaPipe Pose (chest/shoulder landmark motion)
- **Attention metrics** via eye gaze tracking
- **Facial expression analysis** (stress, frustration detection)
- **Workload profiling** (historical load patterns by task type)
- **Integration with calendar/task systems** to correlate load with schedule
- **Mobile app version** for multi-device monitoring
- **Personalized threshold tuning** based on user baseline
- **ML-based scoring** (with user labels for validation)

---

## 10. Deployment & Running

### Prerequisites
- Python 3.10+
- Webcam (laptop or USB)
- Tesseract OCR binary installed
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Installation
```bash
python -m venv venv311
source venv311/bin/activate  # or venv311\Scripts\activate on Windows
pip install -r requirements.txt
```

### Launch
```bash
streamlit run app.py
```

Opens http://localhost:8501 in browser.

---

## 11. Conclusion

The **Cognitive Load Monitor** bridges the gap between productivity metrics and worker wellbeing by providing real-time, actionable feedback on mental fatigue. By combining computer vision, biometric signal processing, and AI-driven interventions, it empowers users to recognize and manage cognitive overload *before* it leads to burnout.

The project demonstrates a full-stack approach:
- **CV Pipeline** extracts meaningful signals from raw video
- **Heuristic scoring** fuses signals into an interpretable metric
- **Regulation layer** triggers timely interventions
- **Interactive dashboard** visualizes trends and provides immediate feedback

This is a foundation for a broader ecosystem of cognitive wellness tools that prioritize the *mental health* of knowledge workers in the digital age.

---

**Team:** 3 parallel workstreams (CV, Detection/AI, Dashboard/UI)  
**Status:** MVP ready for user research and heuristic tuning  
**Deploy:** Streamlit cloud, Docker, or on-premise  
