"""Cognitive Load Monitor — Progressive Interactive Dashboard.

A real-time Streamlit dashboard that visualises webcam-driven cognitive-load
metrics, tracks history over a session, and surfaces actionable solutions.
"""

from __future__ import annotations

import atexit
import datetime
import time
from collections import deque

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import initialize_webcam, release_webcam, run_pipeline
from src.regulation.workflow_regulator import apply_regulation, get_last_recommendation

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

MAX_HISTORY = 600  # keep ~10 min of data


def _status_label(load: float) -> tuple[str, str]:
    """Return (label, colour-hex) based on load score."""
    if load >= 70.0:
        return "Brain Fried", "#e74c3c"
    if load >= 35.0:
        return "Normal", "#f39c12"
    return "Deep Flow", "#2ecc71"


def _load_color(load: float) -> str:
    if load >= 70:
        return "#e74c3c"
    if load >= 35:
        return "#f39c12"
    return "#2ecc71"


def _breathing_color(bpm: float) -> str:
    if 12.0 <= bpm <= 20.0:
        return "#2ecc71"
    return "#e67e22"


def _blink_color(rate: float) -> str:
    if rate <= 15:
        return "#2ecc71"
    if rate <= 30:
        return "#f39c12"
    return "#e74c3c"


# ──────────────────────────────────────────────────────────────
# Session state bootstrap
# ──────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "capture": None,
        "history_ts": deque(maxlen=MAX_HISTORY),
        "history_blink": deque(maxlen=MAX_HISTORY),
        "history_breathing": deque(maxlen=MAX_HISTORY),
        "history_load": deque(maxlen=MAX_HISTORY),
        "history_head": deque(maxlen=MAX_HISTORY),
        "peak_load": 0.0,
        "peak_blink": 0.0,
        "session_start": time.time(),
        "time_deep_flow": 0.0,
        "time_normal": 0.0,
        "time_brain_fried": 0.0,
        "last_tick": time.time(),
        "recommendations": deque(maxlen=20),
        "alert_count": 0,
        "running": True,
        "frame_count": 0,
        "last_toast_ts": 0.0,
        "prev_alerts": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _ensure_capture() -> cv2.VideoCapture:
    cap = st.session_state.get("capture")
    if cap is None or not cap.isOpened():
        cap = initialize_webcam()
        st.session_state.capture = cap
    return cap


def _cleanup_capture() -> None:
    cap = st.session_state.get("capture")
    if cap is not None:
        release_webcam(cap)
        st.session_state.capture = None


atexit.register(_cleanup_capture)

# ──────────────────────────────────────────────────────────────
# Plotly chart builders
# ──────────────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,17,17,0.6)",
    font_color="#c8c8c8",
    margin=dict(l=40, r=20, t=35, b=30),
    height=240,
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
)


def _gauge_chart(value: float, title: str, max_val: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16, "color": "#ddd"}},
        number={"font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "rgba(30,30,30,0.8)",
            "bordercolor": "#333",
            "steps": [
                {"range": [0, max_val * 0.35], "color": "rgba(46,204,113,0.15)"},
                {"range": [max_val * 0.35, max_val * 0.7], "color": "rgba(243,156,18,0.15)"},
                {"range": [max_val * 0.7, max_val], "color": "rgba(231,76,60,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c8c8c8",
        margin=dict(l=30, r=30, t=60, b=20),
        height=220,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sparkline(ts, values, color, title, yrange=None) -> go.Figure:
    fig = go.Figure()
    # Build fill color with alpha
    if "rgb" in color:
        fill_c = color.replace(")", ",0.10)").replace("rgb", "rgba")
    else:
        fill_c = _hex_to_rgba(color, 0.10)
    fig.add_trace(go.Scatter(
        x=list(ts), y=list(values),
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=fill_c,
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    layout = {**_CHART_LAYOUT, "title": {"text": title, "font": {"size": 14}}}
    if yrange:
        layout["yaxis"] = {**layout.get("yaxis", {}), "range": yrange}
    fig.update_layout(**layout)
    return fig


def _zone_pie(deep: float, normal: float, fried: float) -> go.Figure:
    labels = ["Deep Flow", "Normal", "Brain Fried"]
    vals = [max(deep, 0.01), max(normal, 0.01), max(fried, 0.01)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        marker=dict(colors=colors),
        hole=0.55,
        textinfo="percent+label",
        textfont_size=12,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c8c8c8",
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        showlegend=False,
        title={"text": "Time in Zones", "font": {"size": 14}},
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────

_CSS = """
<style>
/* KPI card styling */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,30,30,.85), rgba(50,50,50,.65));
    border-radius: 14px;
    padding: 18px 14px 14px 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: 0 4px 24px rgba(0,0,0,.35);
    transition: transform .15s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-label { font-size: .78rem; color: #aaa; text-transform: uppercase; letter-spacing: .5px; }
.kpi-value { font-size: 1.8rem; font-weight: 700; margin: 2px 0; }
.kpi-sub   { font-size: .72rem; color: #888; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: .4px;
}

/* Recommendation card */
.rec-card {
    background: rgba(44,62,80,.45);
    border-left: 4px solid #3498db;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: .88rem;
    line-height: 1.45;
}
.rec-ts { font-size: .7rem; color: #888; margin-bottom: 2px; }

/* Progress bar */
.prog-track { background: rgba(255,255,255,.08); border-radius: 6px; height: 10px; }
.prog-fill  { border-radius: 6px; height: 10px; transition: width .3s ease; }

/* Scrollable log */
.event-log {
    max-height: 230px;
    overflow-y: auto;
    padding-right: 4px;
}
/* Alert banner */
.alert-banner {
    border-radius: 8px;
    padding: 10px 18px;
    margin: 4px 0;
    font-size: .9rem;
    font-weight: 600;
    line-height: 1.4;
}
.alert-critical {
    background: rgba(231,76,60,.18);
    border-left: 4px solid #e74c3c;
    color: #e74c3c;
}
.alert-warning {
    background: rgba(243,156,18,.15);
    border-left: 4px solid #f39c12;
    color: #f39c12;
}
.alert-ok {
    background: rgba(46,204,113,.12);
    border-left: 4px solid #2ecc71;
    color: #2ecc71;
}
</style>
"""


def _kpi_html(label: str, value: str, color: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────
# Contextual tips (displayed in-dashboard, independent of LLM)
# ──────────────────────────────────────────────────────────────

_TIPS = {
    "brain_fried": [
        "Take a 2-minute deep-breathing break (4 s in, 7 s hold, 8 s out).",
        "Write down your current thought so you won't lose it, then step away.",
        "Drink a glass of water — dehydration amplifies cognitive fatigue.",
        "Switch to ambient / lo-fi music to reduce cortisol.",
        "Stand up and do a 60-second stretch or walk.",
    ],
    "posture": [
        "Move your screen to eye level to reduce forward lean.",
        "Sit back in your chair — your spine will thank you.",
        "Check that your monitor is an arm's length away.",
    ],
    "high_blink": [
        "Look at something 20 ft away for 20 seconds (20-20-20 rule).",
        "Reduce screen brightness or enable a blue-light filter.",
    ],
    "breathing": [
        "Try box breathing: 4 s inhale, 4 s hold, 4 s exhale, 4 s hold.",
        "Slow nasal breathing helps reset your autonomic system.",
    ],
}


def _generate_tips(load: float, head_forward: bool, blink_rate: float, breathing_rate: float) -> list[str]:
    tips: list[str] = []
    if load >= 70:
        tips.append(_TIPS["brain_fried"][int(time.time()) % len(_TIPS["brain_fried"])])
    if head_forward:
        tips.append(_TIPS["posture"][int(time.time()) % len(_TIPS["posture"])])
    if blink_rate > 30:
        tips.append(_TIPS["high_blink"][int(time.time()) % len(_TIPS["high_blink"])])
    if breathing_rate < 12 or breathing_rate > 20:
        tips.append(_TIPS["breathing"][int(time.time()) % len(_TIPS["breathing"])])
    if not tips:
        tips.append("All metrics are in a healthy range — keep it up!")
    return tips


# ──────────────────────────────────────────────────────────────
# Streamlit page config & layout
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cognitive Load Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)
_init_session()
st.markdown(_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Cognitive Load Monitor")
    st.markdown("---")
    elapsed = time.time() - st.session_state.session_start
    st.metric("Session Duration", f"{int(elapsed // 60)}m {int(elapsed % 60)}s")
    st.metric("Alerts Triggered", st.session_state.alert_count)
    st.metric("Peak Load", f"{st.session_state.peak_load:.1f}")
    st.markdown("---")
    st.markdown(
        "**Thresholds**\n"
        "- Deep Flow: load < 35\n"
        "- Normal: 35 <= load < 70\n"
        "- Brain Fried: load >= 70"
    )
    st.markdown("---")
    st.caption("Powered by MediaPipe · OpenAI · Streamlit")

# ── Header ───────────────────────────────────────────────────
header_left, header_right = st.columns([5, 2])
with header_left:
    st.markdown("# AURA (Adaptive User Regulation Assistant)")
    st.caption("Real-time biometric monitoring | Posture tracking | AI-powered rescue plans")
with header_right:
    status_ph = st.empty()

st.markdown("---")

# ── Alert Banner ─────────────────────────────────────────────
alert_banner_ph = st.empty()

# ── Row 1 — KPI Cards ───────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1_ph = kpi1.empty()
kpi2_ph = kpi2.empty()
kpi3_ph = kpi3.empty()
kpi4_ph = kpi4.empty()

# ── Row 2 — Webcam + Gauge ──────────────────────────────────
st.markdown("####")
cam_col, gauge_col = st.columns([3, 2], gap="large")
with cam_col:
    st.markdown("##### Live Webcam Feed")
    frame_ph = st.empty()
with gauge_col:
    st.markdown("##### Load Score")
    gauge_ph = st.empty()
    tips_header_ph = st.empty()
    tips_ph = st.empty()

# ── Row 3 — Time-Series Charts ──────────────────────────────
st.markdown("---")
st.markdown("### Real-Time Trends")
chart1, chart2, chart3 = st.columns(3)
chart_load_ph = chart1.empty()
chart_blink_ph = chart2.empty()
chart_breath_ph = chart3.empty()

# ── Row 4 — Session Analytics + Recommendations ─────────────
st.markdown("---")
st.markdown("### Session Analytics & Solutions")
stats_col, rec_col = st.columns([1, 1], gap="large")

with stats_col:
    st.markdown("##### Zone Distribution")
    zone_pie_ph = st.empty()
    session_stats_ph = st.empty()

with rec_col:
    st.markdown("##### Smart Recommendations")
    rec_list_ph = st.empty()

# ── Row 5 — Event Log ───────────────────────────────────────
st.markdown("---")
with st.expander("Event & Alert Log", expanded=False):
    log_ph = st.empty()

# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

try:
    capture = _ensure_capture()

    while st.session_state.running:
        result = run_pipeline(capture)

        blink_rate = float(result.get("blink_rate", 0.0))
        breathing_rate = float(result.get("breathing_rate", 0.0))
        head_forward = bool(result.get("head_forward", False))
        load_score = float(result.get("load_score", 0.0))

        st.session_state.frame_count += 1
        # Only update heavy charts every ~12 frames (~1 s)
        _update_charts = (st.session_state.frame_count % 12 == 0)

        # Regulation side-effects (alerts, zoom)
        apply_regulation(load_score, head_forward)

        # Check if a new recommendation was generated by the regulator
        last_rec = get_last_recommendation()
        if last_rec and (
            not st.session_state.recommendations
            or st.session_state.recommendations[-1] != last_rec
        ):
            ts_str = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.recommendations.append(f"[{ts_str}] {last_rec}")
            st.session_state.alert_count += 1

        # ── Update history ──
        now = datetime.datetime.now()
        st.session_state.history_ts.append(now)
        st.session_state.history_blink.append(blink_rate)
        st.session_state.history_breathing.append(breathing_rate)
        st.session_state.history_load.append(load_score)
        st.session_state.history_head.append(head_forward)

        # Peaks
        if load_score > st.session_state.peak_load:
            st.session_state.peak_load = load_score
        if blink_rate > st.session_state.peak_blink:
            st.session_state.peak_blink = blink_rate

        # Zone time accounting
        now_t = time.time()
        dt = now_t - st.session_state.last_tick
        st.session_state.last_tick = now_t
        label, color = _status_label(load_score)
        if load_score >= 70:
            st.session_state.time_brain_fried += dt
        elif load_score >= 35:
            st.session_state.time_normal += dt
        else:
            st.session_state.time_deep_flow += dt

        # ── Notifications & Alerts ──
        alerts_html = ""
        active_alerts: set[str] = set()
        now_ts = time.time()

        # High cognitive load
        if load_score >= 70:
            alerts_html += (
                '<div class="alert-banner alert-critical">'
                f'CRITICAL: Cognitive load is {load_score:.1f} (threshold: 70). '
                'Take a break or simplify your current task.</div>'
            )
            active_alerts.add("load_critical")
        elif load_score >= 35:
            alerts_html += (
                '<div class="alert-banner alert-warning">'
                f'WARNING: Cognitive load is {load_score:.1f} (threshold: 35). '
                'Monitor your pace.</div>'
            )
            active_alerts.add("load_warning")

        # High blink rate
        if blink_rate > 30:
            alerts_html += (
                '<div class="alert-banner alert-critical">'
                f'ALERT: Blink rate is {blink_rate:.0f}/min (limit: 30). '
                'Eye strain detected -- look away from the screen.</div>'
            )
            active_alerts.add("blink_high")

        # Breathing out of range
        if breathing_rate < 12 or breathing_rate > 20:
            alerts_html += (
                '<div class="alert-banner alert-warning">'
                f'WARNING: Breathing rate is {breathing_rate:.1f} bpm '
                f'(normal: 12-20). Try slow, controlled breathing.</div>'
            )
            active_alerts.add("breathing_abnormal")

        # Poor posture
        if head_forward:
            alerts_html += (
                '<div class="alert-banner alert-warning">'
                'WARNING: Forward head posture detected. '
                'Sit back and align your screen to eye level.</div>'
            )
            active_alerts.add("posture_forward")

        if alerts_html:
            alert_banner_ph.markdown(alerts_html, unsafe_allow_html=True)
        else:
            alert_banner_ph.markdown(
                '<div class="alert-banner alert-ok">'
                'All clear -- all metrics within healthy limits.</div>',
                unsafe_allow_html=True,
            )

        # Toast notifications (cooldown: 5 s between toasts)
        new_alerts = active_alerts - st.session_state.prev_alerts
        if new_alerts and (now_ts - st.session_state.last_toast_ts) > 5.0:
            for a in new_alerts:
                msg = {
                    "load_critical": f"Cognitive load exceeded 70 ({load_score:.0f})",
                    "load_warning": f"Cognitive load rising ({load_score:.0f})",
                    "blink_high": f"High blink rate: {blink_rate:.0f}/min",
                    "breathing_abnormal": f"Breathing abnormal: {breathing_rate:.1f} bpm",
                    "posture_forward": "Poor posture: head leaning forward",
                }.get(a, "Threshold exceeded")
                st.toast(msg)
                # Log to event log
                ts_str = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.recommendations.append(f"[{ts_str}] ALERT: {msg}")
                st.session_state.alert_count += 1
            st.session_state.last_toast_ts = now_ts
        st.session_state.prev_alerts = active_alerts

        # ── Render status badge ──
        status_ph.markdown(
            f'<div style="text-align:right;margin-top:18px;">'
            f'<span class="status-badge" style="background:{color};color:#fff;">'
            f'{label}</span></div>',
            unsafe_allow_html=True,
        )

        # ── KPI cards ──
        kpi1_ph.markdown(
            _kpi_html("Load Score", f"{load_score:.1f}", _load_color(load_score), "out of 100"),
            unsafe_allow_html=True,
        )
        kpi2_ph.markdown(
            _kpi_html("Blink Rate", f"{blink_rate:.1f}", _blink_color(blink_rate), "blinks / min"),
            unsafe_allow_html=True,
        )
        kpi3_ph.markdown(
            _kpi_html("Breathing", f"{breathing_rate:.1f}", _breathing_color(breathing_rate), "breaths / min"),
            unsafe_allow_html=True,
        )
        posture_txt = "Forward" if head_forward else "Good"
        posture_clr = "#e74c3c" if head_forward else "#2ecc71"
        kpi4_ph.markdown(
            _kpi_html("Posture", posture_txt, posture_clr, "head position"),
            unsafe_allow_html=True,
        )

        # ── Webcam frame ──
        frame = result.get("frame")
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(frame_rgb, channels="RGB", use_container_width=True)

        # ── Gauge (throttled) ──
        _fc = st.session_state.frame_count
        if _update_charts:
            gauge_ph.plotly_chart(
                _gauge_chart(load_score, "Cognitive Load", 100, _load_color(load_score)),
                use_container_width=True,
                key=f"g_{_fc}",
            )

        # ── Tips below gauge ──
        tips = _generate_tips(load_score, head_forward, blink_rate, breathing_rate)
        tips_header_ph.markdown("##### Current Advice")
        tips_html = "".join(f'<div class="rec-card">{t}</div>' for t in tips)
        tips_ph.markdown(tips_html, unsafe_allow_html=True)

        # ── Time-series charts (throttled) ──
        if _update_charts:
            ts_list = list(st.session_state.history_ts)
            if len(ts_list) > 2:
                chart_load_ph.plotly_chart(
                    _sparkline(ts_list, list(st.session_state.history_load), "#e74c3c", "Load Score", [0, 100]),
                    use_container_width=True,
                    key=f"cl_{_fc}",
                )
                chart_blink_ph.plotly_chart(
                    _sparkline(ts_list, list(st.session_state.history_blink), "#3498db", "Blink Rate (bpm)"),
                    use_container_width=True,
                    key=f"cb_{_fc}",
                )
                chart_breath_ph.plotly_chart(
                    _sparkline(ts_list, list(st.session_state.history_breathing), "#2ecc71", "Breathing Rate (bpm)", [5, 30]),
                    use_container_width=True,
                    key=f"cr_{_fc}",
                )

        # ── Zone pie (throttled) ──
        if _update_charts:
            zone_pie_ph.plotly_chart(
                _zone_pie(
                    st.session_state.time_deep_flow,
                    st.session_state.time_normal,
                    st.session_state.time_brain_fried,
                ),
                use_container_width=True,
                key=f"zp_{_fc}",
            )

        # Session stats table (throttled)
        if not _update_charts:
            time.sleep(0.08)
            continue

        avg_load = (
            float(np.mean(list(st.session_state.history_load)))
            if st.session_state.history_load else 0
        )
        avg_blink = (
            float(np.mean(list(st.session_state.history_blink)))
            if st.session_state.history_blink else 0
        )
        avg_breath = (
            float(np.mean(list(st.session_state.history_breathing)))
            if st.session_state.history_breathing else 0
        )
        head_fwd_pct = (
            (sum(st.session_state.history_head) / len(st.session_state.history_head) * 100)
            if st.session_state.history_head else 0
        )

        stats_md = (
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Avg Load Score | **{avg_load:.1f}** |\n"
            f"| Peak Load Score | **{st.session_state.peak_load:.1f}** |\n"
            f"| Avg Blink Rate | **{avg_blink:.1f}** bpm |\n"
            f"| Peak Blink Rate | **{st.session_state.peak_blink:.1f}** bpm |\n"
            f"| Avg Breathing | **{avg_breath:.1f}** bpm |\n"
            f"| Head Forward | **{head_fwd_pct:.0f}%** of session |\n"
            f"| Alerts | **{st.session_state.alert_count}** |\n"
        )
        session_stats_ph.markdown(stats_md)

        # ── Recommendations log ──
        if st.session_state.recommendations:
            recs_html = '<div class="event-log">'
            for r in reversed(st.session_state.recommendations):
                recs_html += f'<div class="rec-card">{r}</div>'
            recs_html += "</div>"
            rec_list_ph.markdown(recs_html, unsafe_allow_html=True)
        else:
            rec_list_ph.info("No AI-generated recommendations yet. They appear when load exceeds 70%.")

        # ── Event log ──
        if st.session_state.recommendations:
            log_ph.markdown(
                "\n\n".join(f"- {r}" for r in reversed(st.session_state.recommendations))
            )
        else:
            log_ph.caption("No events recorded yet.")

        time.sleep(0.08)

except Exception as exc:
    st.error(f"Runtime error: {exc}")
    _cleanup_capture()
