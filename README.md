# 🧠 AURA — Adaptive User Regulation Assistant

> AI-Powered Cognitive Load Detection & Real-Time Mental Regulation System

AURA is an intelligent system that monitors cognitive load in real time using computer vision and biometric proxies.  
It detects mental fatigue, posture strain, and stress patterns, then delivers adaptive interventions to optimize focus and productivity.

---

## 🚀 Overview

Modern professionals and students experience cognitive overload due to prolonged screen exposure, multitasking, and high mental demand.

AURA solves this by:

- 👁 Monitoring blink rate (eye strain detection)
- 🫁 Tracking breathing patterns (stress proxy)
- 🧍 Detecting forward head posture
- 📊 Calculating a dynamic cognitive load score (0–100)
- ⚡ Triggering real-time interventions

---

## 🎯 Key Features

### 🔹 Real-Time Cognitive Monitoring
- Webcam-based facial landmark detection (MediaPipe)
- Eye Aspect Ratio (EAR) blink detection
- Head posture tracking via nose landmark depth
- Simulated breathing rate (extendable to real Pose-based detection)

---

### 🔹 Intelligent Load Scoring Engine
Cognitive Load Score (0–100) calculated using:
- Elevated blink frequency
- Forward head posture
- Abnormal breathing rate

Rule-based scoring model:
- Load < 35 → Deep Flow
- 35–70 → Normal
- > 70 → Brain Fried

---

### 🔹 Smart Workflow Regulation
When thresholds are crossed:
- ⚠ Alert banners appear
- 🔔 Toast notifications trigger
- 🧠 Contextual recommendations displayed
- 🫁 Guided breathing animation activates
- 🔥 Rescue Mode can be triggered

---

### 🔹 Live Analytics Dashboard
- KPI cards
- Real-time trend graphs
- Deep Flow distribution
- Session peak & average stats
- Alert log
- AI recommendations feed

---

## 🏗 System Architecture

