"""
app.py — Clinical Explainable AI for Diabetic Retinopathy Screening
===================================================================
A dashboard for AI-assisted Diabetic Retinopathy screening
with Grad-CAM++ highlighting image regions contributing to the model prediction, Monte Carlo Dropout
uncertainty quantification, and out-of-distribution detection.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torch
import streamlit as st
import sys
import types

# ── Dynamic Package & Path Resolution (Streamlit Cloud Compatibility) ─
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SRC = _ROOT / "src"
if _SRC.exists():
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
else:
    # Fallback if files were uploaded directly to repository root without a src/ folder
    if "src" not in sys.modules:
        _src_mod = types.ModuleType("src")
        _src_mod.__path__ = [str(_ROOT)]
        sys.modules["src"] = _src_mod

from src.config import APP, DATA, PATHS, SCREENING, MODEL, QUALITY, SYSTEM
from src.model import load_model, build_model
from src.predict import predict_pipeline
from src.utils import safe_imread, safe_imwrite
from src.multiframe import run_multiframe_pipeline, extract_frames_from_video

from src.history import (
    save_screening_record,
    get_recent_screenings,
    get_screening_by_id,
    get_all_screenings,
    clear_all_screenings,
)

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaAI • Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Modern Clinical Dark Aesthetic & Glassmorphism ─────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #070B14;
        --card-bg: rgba(15, 23, 42, 0.65);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-cyan: #06B6D4;
        --accent-blue: #3B82F6;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #EF4444;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Streamlit top header & toolbar area override */
    header[data-testid="stHeader"],
    .stApp > header,
    [data-testid="stHeader"],
    .stAppHeader {
        background-color: transparent !important;
        background: transparent !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Streamlit block container padding adjustment */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }

    /* Webkit scrollbar customization */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.4);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.22);
    }

    /* Overall Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(15, 23, 42, 0.98) 0%, #050811 100%);
        color: var(--text-primary);
    }

    /* Top Brand Hero */
    .brand-hero {
        background: linear-gradient(135deg, rgba(20, 32, 60, 0.7) 0%, rgba(10, 16, 32, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 18px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px) saturate(180%);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .brand-hero:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 12px 35px rgba(56, 189, 248, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
        filter: drop-shadow(0 2px 8px rgba(56, 189, 248, 0.2));
    }

    .brand-subtitle {
        color: #CBD5E1;
        font-size: 0.92rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 23, 42, 0.6);
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }

    .badge-online {
        color: #34D399;
        border-color: rgba(52, 211, 153, 0.3);
        background: rgba(6, 78, 59, 0.35);
        filter: drop-shadow(0 0 5px rgba(52, 211, 153, 0.18));
    }

    .badge-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 10px #34D399;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Metric Cards */
    .metric-card-pro {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.35rem 1.15rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .metric-card-pro:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px -4px rgba(56, 189, 248, 0.2), inset 0 1px 0 rgba(255,255,255,0.1);
    }

    /* Custom Metric Colors based on status grades */
    .metric-card-pro.card-grade-0 { border-top: 3px solid #10B981; }
    .metric-card-pro.card-grade-0:hover { border-color: rgba(16, 185, 129, 0.5); box-shadow: 0 12px 24px -4px rgba(16, 185, 129, 0.25); }
    .metric-card-pro.card-grade-1 { border-top: 3px solid #34D399; }
    .metric-card-pro.card-grade-1:hover { border-color: rgba(52, 211, 153, 0.5); box-shadow: 0 12px 24px -4px rgba(52, 211, 153, 0.25); }
    .metric-card-pro.card-grade-2 { border-top: 3px solid #F59E0B; }
    .metric-card-pro.card-grade-2:hover { border-color: rgba(245, 158, 11, 0.5); box-shadow: 0 12px 24px -4px rgba(245, 158, 11, 0.25); }
    .metric-card-pro.card-grade-3 { border-top: 3px solid #F97316; }
    .metric-card-pro.card-grade-3:hover { border-color: rgba(249, 115, 22, 0.5); box-shadow: 0 12px 24px -4px rgba(249, 115, 22, 0.25); }
    .metric-card-pro.card-grade-4 { border-top: 3px solid #EF4444; }
    .metric-card-pro.card-grade-4:hover { border-color: rgba(239, 68, 68, 0.5); box-shadow: 0 12px 24px -4px rgba(239, 68, 68, 0.25); }

    .metric-card-pro.card-blue { border-top: 3px solid #3B82F6; }
    .metric-card-pro.card-blue:hover { border-color: rgba(59, 130, 246, 0.5); box-shadow: 0 12px 24px -4px rgba(59, 130, 246, 0.25); }
    .metric-card-pro.card-purple { border-top: 3px solid #A855F7; }
    .metric-card-pro.card-purple:hover { border-color: rgba(168, 85, 247, 0.5); box-shadow: 0 12px 24px -4px rgba(168, 85, 247, 0.25); }

    .metric-card-pro .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0.25rem 0;
    }

    .metric-card-pro .lbl {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
    }

    .metric-card-pro .sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Recommendation Banners */
    .triage-banner {
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin: 1.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.2rem;
        border: 1px solid;
        animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        backdrop-filter: blur(12px);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .triage-refer {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.35) 0%, rgba(99, 12, 12, 0.55) 100%);
        border-color: rgba(239, 68, 68, 0.35);
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .triage-routine {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.3) 0%, rgba(4, 63, 62, 0.5) 100%);
        border-color: rgba(16, 185, 129, 0.35);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .triage-review {
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.3) 0%, rgba(92, 33, 4, 0.5) 100%);
        border-color: rgba(245, 158, 11, 0.35);
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .triage-title {
        font-size: 1.2rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .triage-desc {
        font-size: 0.88rem;
        color: #E2E8F0;
        margin-top: 0.35rem;
        line-height: 1.4;
    }

    /* Severity Stepper */
    .severity-container {
        display: flex;
        gap: 12px;
        margin: 1.5rem 0;
        background: rgba(13, 20, 38, 0.6);
        padding: 10px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    .severity-step {
        flex: 1;
        padding: 0.9rem 0.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 0.76rem;
        font-weight: 700;
        color: #94A3B8;
        background: rgba(30, 41, 59, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.02);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .severity-step .num-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.05);
        color: #64748B;
        font-size: 0.7rem;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }

    .severity-step.active-0 {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 78, 59, 0.4) 100%);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    .severity-step.active-0 .num-badge {
        background: #10B981;
        color: #0B0F19;
        border-color: #34D399;
        font-weight: 800;
    }

    .severity-step.active-1 {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.18) 0%, rgba(6, 95, 70, 0.4) 100%);
        color: #6EE7B7;
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 15px rgba(52, 211, 153, 0.25);
    }
    .severity-step.active-1 .num-badge {
        background: #34D399;
        color: #0B0F19;
        border-color: #6EE7B7;
        font-weight: 800;
    }

    .severity-step.active-2 {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(120, 53, 15, 0.4) 100%);
        color: #FCD34D;
        border-color: rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }
    .severity-step.active-2 .num-badge {
        background: #F59E0B;
        color: #0B0F19;
        border-color: #FCD34D;
        font-weight: 800;
    }

    .severity-step.active-3 {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.18) 0%, rgba(154, 52, 18, 0.4) 100%);
        color: #FDBA74;
        border-color: rgba(249, 115, 22, 0.4);
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.25);
    }
    .severity-step.active-3 .num-badge {
        background: #F97316;
        color: #0B0F19;
        border-color: #FDBA74;
        font-weight: 800;
    }

    .severity-step.active-4 {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(153, 27, 27, 0.4) 100%);
        color: #FCA5A5;
        border-color: rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
    }
    .severity-step.active-4 .num-badge {
        background: #EF4444;
        color: #0B0F19;
        border-color: #FCA5A5;
        font-weight: 800;
    }

    /* Architecture / Info Card */
    .info-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        background: rgba(15, 23, 42, 0.65);
    }

    .info-card h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    /* Visual Workspace Items */
    .visual-workspace-card {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 10px;
        background: rgba(10, 16, 32, 0.4);
        text-align: center;
        transition: all 0.3s ease;
    }

    .visual-workspace-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        background: rgba(10, 16, 32, 0.6);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    .visual-workspace-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Custom Streamlit Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #070B16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    section[data-testid="stSidebar"] h3 {
        color: #E2E8F0 !important;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    /* Screening History Sidebar Styling */
    .history-group-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #94A3B8;
        margin-top: 0.65rem;
        margin-bottom: 0.35rem;
        padding-left: 0.1rem;
        letter-spacing: 0.02em;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        padding: 0.48rem 0.75rem !important;
        color: #F8FAFC !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        text-align: left !important;
        margin-bottom: 0.3rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(30, 41, 59, 0.9) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:active {
        transform: translateY(0px) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button div[data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button div[data-testid="stMarkdownContainer"] p {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(10, 16, 32, 0.7);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #64748B !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 10px 18px;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #94A3B8 !important;
        background: rgba(255, 255, 255, 0.03);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.25) 0%, rgba(14, 165, 233, 0.25) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.12);
    }

    /* Widget Styling overrides */
    div[data-baseweb="select"] {
        background-color: rgba(13, 20, 38, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"]:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    div[role="listbox"] {
        background-color: #0B0F19 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px;
    }
    div[role="option"] {
        color: #CBD5E1 !important;
    }
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
    }

    /* File Uploader styling */
    div[data-testid="stFileUploader"] {
        background: rgba(13, 20, 38, 0.45);
        border: 2px dashed rgba(56, 189, 248, 0.18);
        border-radius: 14px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.4);
        background: rgba(13, 20, 38, 0.6);
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1D4ED8 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(29, 78, 216, 0.38) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        background: linear-gradient(135deg, #2563EB 0%, #0284C7 100%) !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }

    /* Expanders styling */
    div[data-testid="stExpander"] {
        background: rgba(13, 20, 38, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }
    
    /* Download Button override */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #059669 0%, #059669 100%) !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        box-shadow: 0 8px 18px rgba(5, 150, 105, 0.38) !important;
    }

    /* Radio inputs layout horizontal adjustment */
    div[data-testid="stRadio"] > div {
        gap: 15px;
    }

    /* Clinical Decision Logic Table Styling */
    .decision-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.86rem;
    }
    .decision-table tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: transparent !important;
        background-color: transparent !important;
    }
    .decision-table tr:hover {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    .decision-table td {
        padding: 8px 10px;
        background: transparent !important;
        background-color: transparent !important;
        vertical-align: middle;
        -webkit-user-select: text;
        user-select: text;
    }
    .decision-table td.label-col {
        color: #94A3B8;
        font-weight: 500;
        text-align: left;
    }
    .decision-table td.val-col {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-align: right;
    }
    ::selection {
        background: rgba(56, 189, 248, 0.25);
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Cache Loader ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_model(mtime: float):
    """Load model once and keep in cache, re-loading if checkpoint mtime changes. Fallbacks strictly disabled."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"

    try:
        model = load_model(str(final_model), device)
        is_trained = True
        return model, device, is_trained
    except FileNotFoundError as e:
        st.error("🚨 **RetinaAI Production Model Checkpoint Not Found**")
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.2rem; color: #F8FAFC; margin-bottom: 1.5rem;">
            <h4 style="color: #F87171; margin-top: 0;">Missing Checkpoint: retinaai_efficientnetv2s_deployment.pth</h4>
            <p style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.5;">
                The trained model file (~84 MB) was not found in <code>models/final/</code>. If you uploaded the project via GitHub's browser UI, files over 25 MB are blocked.
            </p>
            <h5 style="color: #38BDF8; margin-bottom: 0.5rem;">Quick Solution for Streamlit Cloud:</h5>
            <ol style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.8;">
                <li>Upload <code>retinaai_efficientnetv2s_deployment.pth</code> to a free repository on <a href="https://huggingface.co/new" target="_blank" style="color: #38BDF8;">Hugging Face Hub</a> or a GitHub Release.</li>
                <li>In your Streamlit Cloud app dashboard, click <b>Settings &rarr; Secrets</b>.</li>
                <li>Add your Hugging Face repository or direct URL:
                    <pre style="background: rgba(15, 23, 42, 0.7); padding: 8px; border-radius: 6px; margin: 6px 0;">HF_MODEL_REPO = "your-hf-username/retinaai-model"\n# or: MODEL_URL = "https://your-direct-link-here..."</pre>
                </li>
                <li>Click <b>Save</b>. Streamlit will automatically download, cache, and initialize the model.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()


_final_model = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"
_mtime = _final_model.stat().st_mtime if _final_model.exists() else 0.0

with st.spinner("Initializing Deep Learning Engine & Weights..."):
    model, device, is_trained = get_cached_model(_mtime)


# ── Helper Plot Functions ───────────────────────────────────────────
def render_probability_chart(probs: list, class_names: list, pred_idx: int, argmax_idx: Optional[int] = None) -> go.Figure:
    """Create a sleek horizontal probability distribution chart."""
    colors = [
        '#10B981',  # No DR (Emerald)
        '#34D399',  # Mild (Green)
        '#F59E0B',  # Moderate (Amber)
        '#F97316',  # Severe (Orange)
        '#EF4444',  # Proliferative (Red)
    ]
    
    # Highlight predicted class with stronger opacity
    opacities = [1.0 if i == pred_idx else 0.45 for i in range(len(probs))]
    
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=class_names,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(color='rgba(255,255,255,0.15)', width=1)
        ),
        text=[f"<b>{p * 100:.1f}%</b>" for p in probs],
        textposition='outside',
        textfont=dict(family="JetBrains Mono, monospace", size=11, color="#F8FAFC"),
        cliponaxis=False,
    ))

    fig.update_layout(
        title=dict(
            text="<b>Multi-Class Softmax Probability Distribution</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#F8FAFC")
        ),
        xaxis=dict(
            title="Probability Confidence (%)",
            range=[0, 115],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            color="#94A3B8",
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=10)
        ),
        yaxis=dict(
            autorange="reversed",
            color="#F8FAFC",
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", size=12)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=50, t=40, b=20),
    )
    return fig


def render_uncertainty_gauge(entropy: float, epistemic: float) -> go.Figure:
    """Create radial uncertainty indicator gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=entropy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Predictive Entropy</b><br><span style='font-size:0.75em;color:#94A3B8'>Total Uncertainty</span>", 'font': {'family': 'Plus Jakarta Sans, sans-serif', 'size': 13, 'color': '#F8FAFC'}},
        number={'suffix': " nats", 'valueformat': ".3f", 'font': {'family': 'JetBrains Mono, monospace', 'size': 20, 'color': '#38BDF8'}},
        gauge={
            'axis': {'range': [0, 1.61], 'tickcolor': "#64748B", 'tickfont': {'family': 'JetBrains Mono, monospace', 'size': 10, 'color': '#64748B'}},
            'bar': {'color': "#06B6D4", 'thickness': 0.3},
            'bgcolor': "rgba(15, 23, 42, 0.4)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.08)",
            'steps': [
                {'range': [0, 0.6], 'color': "rgba(16, 185, 129, 0.25)"},
                {'range': [0.6, SCREENING.uncertainty_refer_threshold], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [SCREENING.uncertainty_refer_threshold, 1.61], 'color': "rgba(239, 68, 68, 0.25)"}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.8,
                'value': SCREENING.uncertainty_refer_threshold
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=190,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ── Top Brand Header ────────────────────────────────────────────────
st.markdown(f"""
<div class="brand-hero">
    <div>
        <h1 class="brand-title">RetinaAI • Explainable DR Screening</h1>
        <div class="brand-subtitle">Clinical Deep Learning Screening Suite with Grad-CAM++ Attention & Monte Carlo Dropout</div>
    </div>
    <div style="display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;">
        <div class="badge-pill badge-online">
            <span class="badge-pulse"></span>
            {'MODEL LOADED · HASH VERIFIED' if is_trained else 'Prototype Mode'}
        </div>
        <div class="badge-pill" style="color: #93C5FD; border-color: rgba(147, 197, 253, 0.2);">
            ⚙️ {MODEL.backbone.split('.')[0]}
        </div>
        <div class="badge-pill" style="color: #F472B6; border-color: rgba(244, 114, 182, 0.2);">
            ⚡ {str(device).upper()}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Controls ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📥 Patient Fundus Input")

    input_mode = st.radio(
        "Source Method",
        ["📁 Upload Single Fundus", "📱 Smartphone Multi-Frame", "🧪 Sample Patient Cases"],
        horizontal=True,
        label_visibility="collapsed"
    )

    image_to_process_path = None
    uploaded_image_bytes = None
    multiframe_bytes_list = None
    sample_name = ""

    if input_mode == "📁 Upload Single Fundus":
        uploaded_file = st.file_uploader(
            "Select retinal fundus photograph",
            type=APP.supported_formats,
            help="Supports high-resolution PNG, JPG, JPEG, and TIFF fundus images."
        )
        if uploaded_file is not None:
            uploaded_image_bytes = uploaded_file.getvalue()

    elif input_mode == "📱 Smartphone Multi-Frame":
        st.markdown("<div style='font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.5rem;'>Upload a burst sequence of <b>8–10 frames</b> or a short video clip from a smartphone fundus adapter.</div>", unsafe_allow_html=True)
        uploaded_multi = st.file_uploader(
            "Upload 8–10 burst frames or video clip",
            type=["png", "jpg", "jpeg", "mp4", "mov", "avi"],
            accept_multiple_files=True,
            help="Select 8-10 consecutive frames or a video recording (.mp4/.mov)."
        )
        if uploaded_multi:
            if len(uploaded_multi) == 1 and uploaded_multi[0].name.lower().endswith((".mp4", ".mov", ".avi")):
                _extracted = extract_frames_from_video(uploaded_multi[0].getvalue(), max_frames=10)
                multiframe_bytes_list = [f["image_bytes"] for f in _extracted]
                st.session_state["_mf_is_video"] = True
            else:
                multiframe_bytes_list = [f.getvalue() for f in uploaded_multi]
                st.session_state["_mf_is_video"] = False
            st.success(f"Loaded {len(multiframe_bytes_list)} frames for multi-frame analysis")

    else:
        # Check if we have sample images in data/raw/train_images
        sample_dir = PATHS.data_raw / "train_images"
        sample_files = list(sample_dir.glob("*.png")) if sample_dir.exists() else []

        if sample_files:
            sample_options = {
                "0e0003ddd8df": "🟢 Grade 0: Normal Retina (No DR)",
                "33ffddea8c6e": "🟡 Grade 1: Mild NPDR",
                "188a9323be03": "🟠 Grade 2: Moderate NPDR",
                "8201cab8322d": "🔴 Grade 3: Severe NPDR",
                "37c4dfe03aba": "🟣 Grade 4: Proliferative DR",
            }
            # Filter available on disk
            available_samples = [s for s in sample_options if (sample_dir / f"{s}.png").exists()]
            if not available_samples:
                available_samples = [f.stem for f in sample_files[:6]]
                sample_labels = {s: f"Patient Case {s[:8]}..." for s in available_samples}
            else:
                sample_labels = sample_options

            selected_sample = st.selectbox(
                "Select Sample Case",
                available_samples,
                format_func=lambda k: sample_labels.get(k, k)
            )
            if selected_sample:
                sample_file_path = sample_dir / f"{selected_sample}.png"
                if sample_file_path.exists():
                    image_to_process_path = str(sample_file_path)
                    sample_name = sample_labels.get(selected_sample, selected_sample)
        else:
            st.info("No preloaded sample images found. Please upload an image.")

    st.markdown("---")

    # ── Screening History Section ───────────────────────────────────
    st.markdown("### 🕘 Screening History")

    history_groups = get_recent_screenings(limit=15)
    total_screenings = sum(len(items) for items in history_groups.values())

    if total_screenings == 0:
        st.markdown("""
        <div style="padding: 0.6rem 0.2rem; color: #94A3B8; font-size: 0.82rem; line-height: 1.5;">
            <div style="font-weight: 600; color: #CBD5E1; margin-bottom: 0.2rem;">No previous screenings yet.</div>
            <div style="color: #64748B; font-size: 0.78rem;">Completed screenings will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for group_name in ["Today", "Yesterday", "Earlier"]:
            items = history_groups.get(group_name, [])
            if items:
                st.markdown(f"<div class='history-group-title'>{group_name}</div>", unsafe_allow_html=True)
                for item in items:
                    dec = item.get("decision", "")
                    if "RETAKE" in dec or "REJECT" in dec or not item.get("quality_passed", 1):
                        t_chip = "RETAKE"
                    elif "REFER" in dec:
                        t_chip = "REFER"
                    else:
                        t_chip = "ROUTINE"
                    btn_label = f"#{item['id']} {item['emoji']} {t_chip} · Grade {item['dr_stage']} · {item['time_display']} ›"
                    if st.button(btn_label, key=f"hist_btn_{item['id']}", use_container_width=True):
                        st.session_state["selected_history_id"] = item["id"]
                        st.session_state["view_all_history"] = False
                        st.rerun()

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("📋 View All History", key="view_all_history_btn", use_container_width=True):
            st.session_state["view_all_history"] = True
            st.session_state["selected_history_id"] = None
            st.rerun()

        # Clear history with confirmation
        if st.session_state.get("confirm_clear_history", False):
            st.markdown("""
            <div style="margin-top: 0.4rem; padding: 0.6rem; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px;">
                <div style="color: #FCA5A5; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.4rem;">⚠️ Clear all screening records?</div>
            </div>
            """, unsafe_allow_html=True)
            col_clear_yes, col_clear_no = st.columns(2)
            with col_clear_yes:
                if st.button("✓ Yes, Clear", key="btn_confirm_clear_yes", use_container_width=True):
                    clear_all_screenings()
                    st.session_state["selected_history_id"] = None
                    st.session_state["confirm_clear_history"] = False
                    st.session_state["last_saved_run_fingerprint"] = None
                    st.rerun()
            with col_clear_no:
                if st.button("Cancel", key="btn_confirm_clear_no", use_container_width=True):
                    st.session_state["confirm_clear_history"] = False
                    st.rerun()
        else:
            if st.button("🗑 Clear History", key="btn_clear_history_trigger", use_container_width=True):
                st.session_state["confirm_clear_history"] = True
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Screening Telemetry")
    
    with st.expander("System Specs & Thresholds", expanded=False):
        st.markdown(f"""
        - **Input Resolution**: `{DATA.image_size} × {DATA.image_size}`
        - **MC Dropout Passes**: `{MODEL.mc_dropout_passes}` stochastic iterations
        - **Quality Blur Min**: `{QUALITY.min_blur_score}`
        - **Referral Threshold**: `{SCREENING.high_confidence_threshold * 100:.0f}%`
        - **Uncertainty Limit**: `{SCREENING.uncertainty_refer_threshold} nats`
        """)

    st.markdown("### 📋 Severity Index")
    st.caption("""
    - **Grade 0**: No Diabetic Retinopathy
    - **Grade 1**: Mild NPDR (Microaneurysms)
    - **Grade 2**: Moderate NPDR (Hemorrhages/Exudates)
    - **Grade 3**: Severe NPDR 
    - **Grade 4**: Proliferative DR (Neovascularization)
    """)


# ── Main Processing Flow ────────────────────────────────────────────
is_multiframe = (input_mode == "📱 Smartphone Multi-Frame" and multiframe_bytes_list is not None and len(multiframe_bytes_list) > 0)
has_image = (uploaded_image_bytes is not None) or (image_to_process_path is not None) or is_multiframe

# Check if viewing all history
if st.session_state.get("view_all_history"):
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.6rem;">📋</span>
            <div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC;">All Screening Records</div>
                <div style="font-size: 0.82rem; color: #94A3B8;">Historical log of completed AI-assisted retinal screenings</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Live Screening", key="btn_back_from_all_history"):
        st.session_state["view_all_history"] = False
        st.rerun()

    all_records = get_all_screenings()
    if not all_records:
        st.info("No screening records stored yet.")
    else:
        df_display = pd.DataFrame([
            {
                "ID": f"#{r['id']}",
                "Date & Time": r["formatted_time"],
                "Image": r["image_name"],
                "DR Result": f"{r.get('emoji', '⚪')} Grade {r['dr_stage']} ({r['dr_stage_name']})",
                "Confidence": f"{r['dr_confidence'] * 100:.1f}%",
                "Quality": r["image_quality"],
                "Action / Recommendation": r["decision"],
            }
            for r in all_records
        ])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.caption("Select any past record from the sidebar to inspect its detailed screening report.")

# Check if a specific historical record was selected
elif st.session_state.get("selected_history_id"):
    hist_record = get_screening_by_id(st.session_state["selected_history_id"])
    if not hist_record:
        st.warning("Selected historical screening record could not be found.")
        if st.button("← Return to Live Screening", key="btn_return_missing_hist"):
            st.session_state["selected_history_id"] = None
            st.rerun()
    else:
        # Render historical result
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.6rem;">📋</span>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC;">Previous Screening Result</div>
                    <div style="font-size: 0.82rem; color: #94A3B8;">Historical record from <b>{hist_record['formatted_datetime']}</b> • Scan: <b>{hist_record['image_name']}</b></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Return to Live Screening", key="btn_exit_hist_view"):
            st.session_state["selected_history_id"] = None
            st.rerun()

        h_pred_idx = hist_record["dr_stage"]
        h_pred_name = hist_record["dr_stage_name"]
        h_conf = hist_record["dr_confidence"]
        h_decision = hist_record["decision"]
        h_uncert = hist_record["uncertainty"]
        h_ood = hist_record["ood_score"]
        h_quality = hist_record["image_quality"]
        h_quality_passed = bool(hist_record["quality_passed"])

        # Historical Triage Banner
        if "REFER" in h_decision and "REVIEW" not in h_decision:
            t_class = "triage-refer"
            t_icon = "🚨"
            t_badge = "SPECIALIST OPHTHALMOLOGY REFERRAL REQUIRED"
            t_action = "Referral recommended for clinical slit-lamp / OCT exam due to detectable diabetic retinopathy or high risk."
        elif "ROUTINE" in h_decision or "NO REFERRAL" in h_decision:
            t_class = "triage-routine"
            t_icon = "✅"
            t_badge = "ROUTINE ANNUAL RESCREENING RECOMMENDED"
            t_action = "No vision-threatening diabetic retinopathy signs detected. Schedule standard 12-month follow-up."
        else:
            t_class = "triage-review"
            t_icon = "⚠️"
            t_badge = "CLINICAL MANUAL TRIAGE REVIEW RECOMMENDED"
            t_action = "Borderline confidence or slight uncertainty detected. Secondary human clinician review advised."

        st.markdown(f"""
        <div class="triage-banner {t_class}">
            <div>
                <div class="triage-title">
                    <span>{t_icon}</span>
                    <span>{t_badge}</span>
                </div>
                <div class="triage-desc">{t_action}</div>
            </div>
            <div style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 0.9rem; padding: 0.4rem 0.8rem; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                ACTION: {h_decision}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stage Tracker Bar
        st.markdown(f"""
        <div class="severity-container">
            <div class="severity-step {'active-0' if h_pred_idx == 0 else ''}">
                <div class="num-badge">0</div>
                <div style="font-weight: 800;">GRADE 0</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">No DR</div>
            </div>
            <div class="severity-step {'active-1' if h_pred_idx == 1 else ''}">
                <div class="num-badge">1</div>
                <div style="font-weight: 800;">GRADE 1</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Mild NPDR</div>
            </div>
            <div class="severity-step {'active-2' if h_pred_idx == 2 else ''}">
                <div class="num-badge">2</div>
                <div style="font-weight: 800;">GRADE 2</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Mod NPDR</div>
            </div>
            <div class="severity-step {'active-3' if h_pred_idx == 3 else ''}">
                <div class="num-badge">3</div>
                <div style="font-weight: 800;">GRADE 3</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Sev NPDR</div>
            </div>
            <div class="severity-step {'active-4' if h_pred_idx == 4 else ''}">
                <div class="num-badge">4</div>
                <div style="font-weight: 800;">GRADE 4</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Prolif DR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # High-Level Metric Tiles
        hm_col1, hm_col2, hm_col3, hm_col4, hm_col5 = st.columns(5)
        stage_colors = ["#10B981", "#34D399", "#F59E0B", "#F97316", "#EF4444"]

        with hm_col1:
            st.markdown(f"""
            <div class="metric-card-pro card-grade-{h_pred_idx}">
                <div class="lbl">Predicted Stage</div>
                <div class="val" style="color: {stage_colors[h_pred_idx]};">Grade {h_pred_idx}</div>
                <div class="sub">{h_pred_name}</div>
            </div>
            """, unsafe_allow_html=True)

        with hm_col2:
            st.markdown(f"""
            <div class="metric-card-pro card-blue">
                <div class="lbl">Model Confidence</div>
                <div class="val" style="color: #60A5FA;">{h_conf * 100:.1f}%</div>
                <div class="sub">Softmax Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

        with hm_col3:
            st.markdown(f"""
            <div class="metric-card-pro card-emerald">
                <div class="lbl">Quality Gate</div>
                <div class="val" style="color: {'#34D399' if h_quality_passed else '#EF4444'};">{'PASS' if h_quality_passed else 'REJECT'}</div>
                <div class="sub">{h_quality}</div>
            </div>
            """, unsafe_allow_html=True)

        with hm_col4:
            st.markdown(f"""
            <div class="metric-card-pro card-purple">
                <div class="lbl">Predictive Uncertainty</div>
                <div class="val" style="color: #C084FC;">{h_uncert:.4f}</div>
                <div class="sub">Epistemic Entropy</div>
            </div>
            """, unsafe_allow_html=True)

        with hm_col5:
            st.markdown(f"""
            <div class="metric-card-pro card-amber">
                <div class="lbl">OOD Detection</div>
                <div class="val" style="color: #FBBF24;">{h_ood:.1f}</div>
                <div class="sub">Free Energy Score</div>
            </div>
            """, unsafe_allow_html=True)

        # Formatted Clinical Screening Record Card
        st.markdown("---")
        st.markdown("### 📄 Formatted Clinical Screening Record (Historical)")
        
        hist_summary_text = f"""
================================================================================
              RETINAAI • HISTORICAL CLINICAL SCREENING RECORD
================================================================================
RECORD ID: #{hist_record['id']}
PATIENT SCAN: {hist_record['image_name']}
SCREENING DATE: {hist_record['formatted_datetime']}
--------------------------------------------------------------------------------
FINDINGS & CLASSIFICATION:
  • Predicted Stage: Grade {h_pred_idx} — {h_pred_name}
  • Confidence Score: {h_conf * 100:.1f}%
  • Referable Status: {'REFERABLE (Stage ≥ 2)' if h_pred_idx in SCREENING.referable_grades else 'NON-REFERABLE'}
  • Epistemic Uncertainty: {h_uncert:.4f} nats
  • Quality Verification: {h_quality}
--------------------------------------------------------------------------------
TRIAGE RECOMMENDATION:
  {h_decision}
--------------------------------------------------------------------------------
DISCLAIMER: AI-generated screening assistance tool for clinical research.
            Final diagnosis requires confirmation by a licensed ophthalmologist.
================================================================================
"""
        st.code(hist_summary_text, language="text")
        st.download_button(
            "📥 Download Historical Clinical Summary (.txt)",
            data=hist_summary_text,
            file_name=f"RetinaAI_Historical_Grade{h_pred_idx}_{hist_record['id']}.txt",
            mime="text/plain",
            key=f"btn_download_hist_{hist_record['id']}"
        )

elif has_image:
    temp_file_to_clean = None
    if is_multiframe:
        with st.spinner("📱 Ingesting Smartphone Burst, Assessing Frame Quality & Performing Bayesian Multi-Frame Fusion..."):
            start_time = time.time()
            try:
                report = run_multiframe_pipeline(multiframe_bytes_list, model, device=device, is_video=st.session_state.get("_mf_is_video", False))
                elapsed_time = time.time() - start_time
                success = True
            except Exception as e:
                st.error(f"Multi-Frame Pipeline Error: {str(e)}")
                success = False
    else:
        if uploaded_image_bytes is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_image_bytes)
                image_to_process_path = tmp.name
                temp_file_to_clean = tmp.name

        # Run complete single-image inference pipeline
        with st.spinner("🔬 Retinal Quality Gate → Preprocessing → Inference → Uncertainty → Grad-CAM++ Explanation..."):
            start_time = time.time()
            try:
                report = predict_pipeline(image_to_process_path, model, device=device)
                elapsed_time = time.time() - start_time
                success = True
            except Exception as e:
                st.error(f"Inference Pipeline Error: {str(e)}")
                success = False

    # Clean temporary file
    if temp_file_to_clean and os.path.exists(temp_file_to_clean):
        try:
            os.remove(temp_file_to_clean)
        except Exception:
            pass

    if success:
        if not report.get("is_fundus", True):
            domain_val = report.get("domain_validation", {})
            reasons = domain_val.get("reasons", ["Input image does not match retinal fundus characteristics."])

            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(153, 27, 27, 0.45) 0%, rgba(99, 12, 12, 0.65) 100%); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 16px; padding: 1.5rem 1.8rem; margin: 1.2rem 0;">
                <div style="font-size: 1.35rem; font-weight: 800; color: #FCA5A5; display: flex; align-items: center; gap: 0.6rem;">
                    <span>🚫</span> INVALID INPUT — RETINAL FUNDUS IMAGE REQUIRED
                </div>
                <div style="color: #F8FAFC; font-size: 0.92rem; margin-top: 0.4rem; line-height: 1.5;">
                    RetinaAI accepts retinal fundus images only. Non-retinal photographs (such as fruits, objects, selfies, documents, or screenshots) are automatically rejected before classification to prevent silent misdiagnosis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_rej1, col_rej2 = st.columns([1, 1.2])
            with col_rej1:
                st.markdown('<div class="visual-workspace-card"><div class="visual-workspace-title">Uploaded Non-Fundus Input</div>', unsafe_allow_html=True)
                st.image(report["visuals"]["original"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_rej2:
                reasons_html = "".join([f"<li style='margin-bottom:6px;'>❌ {r}</li>" for r in reasons])
                st.markdown(f"""
                <div class="info-card">
                    <h4 style="color: #EF4444; margin-top: 0; margin-bottom: 0.6rem;">Domain Validation Rejection Reasons</h4>
                    <ul style="color: #FCA5A5; font-size: 0.88rem; padding-left: 1.2rem; line-height: 1.6;">
                        {reasons_html}
                    </ul>
                    <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 0.85rem; color: #94A3B8; border: 1px solid rgba(255,255,255,0.08);">
                        💡 <b>Action Required:</b> Please upload a valid retinal fundus photograph showing the optic disc, retinal vessels, and macula.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.stop()

        pred = report["prediction"]
        uncert = report["uncertainty"]
        qual = report.get("quality") or (report.get("per_frame_reports", [{}])[0].get("quality", {"passed": True, "blur_score": 85.0, "reasons": []}))
        ood = report["ood"]
        decision = report["decision"]
        pred_idx = pred["class_index"]
        pred_name = pred["class_name"]
        conf = pred["confidence"]
        argmax_idx = pred.get("argmax_class", pred_idx)
        argmax_name = pred.get("argmax_name", pred_name)
        argmax_conf = pred.get("argmax_confidence", conf)
        is_adjusted = pred.get("is_ordinal_adjusted", False)
        exp_score = pred.get("expected_score", float(pred_idx))



        # Deduplicated history persistence
        scan_title = sample_name if sample_name else (uploaded_file.name if (input_mode == "📁 Upload Single Fundus" and uploaded_file is not None) else ("Smartphone Multi-Frame Burst" if is_multiframe else "Retinal Fundus Scan"))
        current_fingerprint = f"{scan_title}_{pred_idx}_{conf:.4f}"
        if st.session_state.get("last_saved_run_fingerprint") != current_fingerprint:
            blur_val = qual.get("blur_score", 0.0)
            q_str = f"{'Passed' if qual.get('passed', True) else 'Failed'} (Score: {blur_val:.1f})"
            save_screening_record(
                image_name=scan_title,
                dr_stage=pred_idx,
                dr_stage_name=pred_name,
                dr_confidence=conf,
                image_quality=q_str,
                quality_passed=bool(qual.get("passed", True)),
                uncertainty=float(uncert.get("epistemic", uncert.get("entropy", 0.0))),
                ood_score=float(ood.get("energy_score", 0.0)),
                decision=decision,
                metadata={
                    "all_probabilities": pred.get("all_probabilities", []),
                    "decision": decision,
                    "model_backbone": MODEL.backbone,
                }
            )
            st.session_state["last_saved_run_fingerprint"] = current_fingerprint

        # ── Stage Tracker Bar ───────────────────────────────────────
        st.markdown(f"""
        <div class="severity-container">
            <div class="severity-step {'active-0' if pred_idx == 0 else ''}">
                <div class="num-badge">0</div>
                <div style="font-weight: 800;">GRADE 0</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">No DR</div>
            </div>
            <div class="severity-step {'active-1' if pred_idx == 1 else ''}">
                <div class="num-badge">1</div>
                <div style="font-weight: 800;">GRADE 1</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Mild NPDR</div>
            </div>
            <div class="severity-step {'active-2' if pred_idx == 2 else ''}">
                <div class="num-badge">2</div>
                <div style="font-weight: 800;">GRADE 2</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Mod NPDR</div>
            </div>
            <div class="severity-step {'active-3' if pred_idx == 3 else ''}">
                <div class="num-badge">3</div>
                <div style="font-weight: 800;">GRADE 3</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Sev NPDR</div>
            </div>
            <div class="severity-step {'active-4' if pred_idx == 4 else ''}">
                <div class="num-badge">4</div>
                <div style="font-weight: 800;">GRADE 4</div>
                <div style="font-size: 0.65rem; opacity: 0.8; margin-top: 2px;">Prolif DR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── High-Level Metric Tiles ─────────────────────────────────
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            stage_colors = ["#10B981", "#34D399", "#F59E0B", "#F97316", "#EF4444"]
            dim_style = 'style="opacity: 0.55;"' if not qual.get('passed', True) else ''
            sub_label = f"{pred_name} (Unreliable)" if not qual.get('passed', True) else pred_name
            st.markdown(f"""
            <div class="metric-card-pro card-grade-{pred_idx}" {dim_style}>
                <div class="lbl">PREDICTED STAGE</div>
                <div class="val" style="color: {stage_colors[pred_idx]};">Grade {pred_idx}</div>
                <div class="sub">{sub_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-card-pro card-blue">
                <div class="lbl">MODEL CONFIDENCE</div>
                <div class="val" style="color: #60A5FA;">{conf * 100:.1f}%</div>
                <div class="sub">Softmax Likelihood</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-card-pro card-grade-0">
                <div class="lbl">BAYESIAN ENTROPY</div>
                <div class="val" style="color: #34D399;">{uncert['entropy']:.3f}</div>
                <div class="sub">MC Uncertainty (nats)</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            qual_grade = 'grade-0' if qual['passed'] else 'grade-4'
            qual_colors = {'grade-0': '#34D399', 'grade-4': '#F87171'}
            st.markdown(f"""
            <div class="metric-card-pro card-{qual_grade}">
                <div class="lbl">IMAGE QUALITY</div>
                <div class="val" style="color: {qual_colors[qual_grade]};">{'PASSED' if qual['passed'] else 'FLAGGED'}</div>
                <div class="sub">Sharpness: {qual['blur_score']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col5:
            st.markdown(f"""
            <div class="metric-card-pro card-purple">
                <div class="lbl">INFERENCE LATENCY</div>
                <div class="val" style="color: #C084FC;">{elapsed_time:.2f}s</div>
                <div class="sub">{MODEL.mc_dropout_passes}x Forward Passes</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabbed Clinical Insights ────────────────────────────────
        tab_diag, tab_xai, tab_uncert, tab_qual, tab_summary = st.tabs([
            "📊 Diagnostic Breakdown",
            "👁️ Grad-CAM++ Localization",
            "🧠 Uncertainty & Out-of-Distribution",
            "🔍 Quality Gate & Optics",
            "📄 Clinical Report Summary"
        ])

        # TAB 1: DIAGNOSTIC BREAKDOWN
        with tab_diag:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                prob_fig = render_probability_chart(pred["all_probabilities"], DATA.class_names, pred_idx)
                st.plotly_chart(prob_fig, use_container_width=True)

            with c2:
                entropy_val = uncert.get("entropy", 0.0)
                epistemic_val = uncert.get("epistemic", 0.0)
                ood_val = ood.get("energy_score", 0.0)

                if not qual.get("passed", True):
                    referable_str = "Unreliable (low quality)"
                    referable_clr = "#FCD34D"
                elif pred_idx in SCREENING.referable_grades:
                    referable_str = "YES (Stage ≥ 2)"
                    referable_clr = "#EF4444"
                else:
                    referable_str = "NO (Stage < 2)"
                    referable_clr = "#10B981"

                st.markdown("#### 🩺 Clinical Decision Logic")
                st.markdown(f"""
                <div class="info-card">
                    <table class="decision-table">
                        <tr>
                            <td class="label-col">Quality Gate Check:</td>
                            <td class="val-col" style="color: {'#34D399' if qual['passed'] else '#EF4444'};">
                                {'✓ Passed All Filters' if qual['passed'] else '✗ Rejected'}
                            </td>
                        </tr>
                        <tr>
                            <td class="label-col">Dominant Severity:</td>
                            <td class="val-col" style="color: #F8FAFC;">
                                Grade {pred_idx} ({pred_name})
                            </td>
                        </tr>
                        <tr>
                            <td class="label-col">Referable DR Status:</td>
                            <td class="val-col" style="color: {referable_clr};">
                                {referable_str}
                            </td>
                        </tr>
                        <tr>
                            <td class="label-col">Epistemic Mutual Info:</td>
                            <td class="val-col" style="color: #38BDF8;">
                                {epistemic_val:.4f} nats
                            </td>
                        </tr>
                        <tr>
                            <td class="label-col">OOD Energy Index:</td>
                            <td class="val-col" style="color: #A78BFA;">
                                {ood_val:.2f}
                            </td>
                        </tr>
                    </table>
                    <div style="margin-top: 0.75rem; font-size: 0.74rem; color: #64748B; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.45rem;">
                        ℹ️ Automated screening aid for clinical workflow triage. Final diagnosis requires specialist evaluation.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if report.get("mode") == "multiframe":
                st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
                st.markdown("#### 📱 Smartphone Multi-Frame Analysis")
                tele = report.get("telemetry", {})

                # ── Frame Count Metrics ──
                mf_c1, mf_c2, mf_c3, mf_c4, mf_c5 = st.columns(5)
                with mf_c1:
                    st.metric("Total Frames Captured", f"{tele.get('total_captured', 0)}")
                with mf_c2:
                    st.metric("Usable Frames", f"{tele.get('usable_frames', 0)}")
                with mf_c3:
                    st.metric("Rejected Frames", f"{tele.get('rejected_frames', 0)}")
                with mf_c4:
                    st.metric("Analyzed Frames", f"{tele.get('analyzed_frames', 0)}")
                with mf_c5:
                    st.metric("Best Frame", f"Frame #{report.get('exemplar_index', 0)}")

                # ── Selected Frame Indices ──
                selected_indices = tele.get('selected_frame_indices', [])
                if selected_indices:
                    idx_str = ", ".join([f"Frame {idx}" for idx in selected_indices])
                    st.markdown(f"""
                    <div class="info-card" style="margin-top: 0.8rem;">
                        <b style="color: #38BDF8;">🎯 Frames Considered by Model:</b>
                        <span style="font-family: 'JetBrains Mono'; color: #F8FAFC; margin-left: 0.5rem;">{idx_str}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Per-Frame Detail Table (all frames) ──
                with st.expander("🔍 All Frames — Quality & Prediction Breakdown", expanded=True):
                    all_meta = report.get("all_frame_meta", [])
                    rows = []
                    for fm in all_meta:
                        pred_info = fm.get("prediction")
                        row = {
                            "Frame": fm["frame_index"],
                            "Time (s)": f"{fm['timestamp']:.2f}" if fm["timestamp"] > 0 else "—",
                            "Quality": f"{fm['quality_score']:.1f}",
                            "Status": fm["quality_status"],
                            "Prediction": f"Grade {pred_info['class_index']} ({pred_info['class_name']})" if pred_info else "—",
                            "Confidence": f"{pred_info['confidence'] * 100:.1f}%" if pred_info else "—",
                        }
                        rows.append(row)
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # ── Per-Frame Probabilities (analyzed only) ──
                with st.expander("📊 Per-Frame Softmax Probabilities & Fusion Weights", expanded=False):
                    per_frame = report.get("per_frame_reports", [])
                    weights = report.get("weights", [])
                    prob_rows = []
                    for i, pf in enumerate(per_frame):
                        probs = pf["prediction"]["all_probabilities"]
                        prob_rows.append({
                            "Frame": pf["frame_index"],
                            "P(No DR)": f"{probs[0]*100:.1f}%",
                            "P(Mild)": f"{probs[1]*100:.1f}%",
                            "P(Moderate)": f"{probs[2]*100:.1f}%",
                            "P(Severe)": f"{probs[3]*100:.1f}%",
                            "P(PDR)": f"{probs[4]*100:.1f}%",
                            "Weight": f"{weights[i]*100:.1f}%" if i < len(weights) else "—",
                        })
                    if prob_rows:
                        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

                # ── Inter-Frame Disagreement ──
                disagreement = tele.get('inter_frame_disagreement', 0.0)
                st.markdown(f"""
                <div class="info-card" style="margin-top: 0.8rem;">
                    <b style="color: #FBBF24;">📐 Inter-Frame Disagreement:</b>
                    <span style="font-family: 'JetBrains Mono'; color: #F8FAFC; margin-left: 0.5rem;">{disagreement:.4f} std</span>
                    <span style="color: #64748B; font-size: 0.8rem; margin-left: 0.5rem;">({'Low — Consistent' if disagreement < 0.1 else 'Moderate' if disagreement < 0.2 else 'High — Review advised'})</span>
                </div>
                """, unsafe_allow_html=True)

        # TAB 2: EXPLAINABILITY & GRAD-CAM++
        with tab_xai:
            st.markdown("### 👁️ Visual Explainability (Grad-CAM++ Attention)")
            st.caption("Highlights the regions that most influenced the prediction. This is model attention, not a confirmed lesion detector.")

            col_img1, col_img2, col_img3, col_img4 = st.columns(4)

            with col_img1:
                st.markdown('<div class="visual-workspace-card"><div class="visual-workspace-title">1. Raw Fundus Scan</div>', unsafe_allow_html=True)
                st.image(report["visuals"]["original"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_img2:
                st.markdown('<div class="visual-workspace-card"><div class="visual-workspace-title">2. CLAHE & Color Normalized</div>', unsafe_allow_html=True)
                st.image(report["visuals"]["preprocessed"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_img3:
                st.markdown('<div class="visual-workspace-card"><div class="visual-workspace-title">3. Grad-CAM++ Attention Heatmap</div>', unsafe_allow_html=True)
                # Convert grayscale heatmap to RGB colormap for preview
                heat = report["visuals"]["heatmap"]
                heat_colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
                st.image(heat_colored, use_container_width=True)
                st.markdown("""
                <div style="height: 6px; border-radius: 3px; background: linear-gradient(90deg, #000080, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000, #800000); margin-top: 6px; margin-bottom: 2px;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748B;"><span>Low attention</span><span>High attention</span></div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_img4:
                st.markdown(f'<div class="visual-workspace-card"><div class="visual-workspace-title">4. Attention Overlay (predicted: Grade {pred_idx})</div>', unsafe_allow_html=True)
                st.image(report["visuals"]["overlay"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card" style="margin-top: 1rem;">
                <b style="color: #38BDF8;">💡 Interpreting Attention Maps:</b>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.3rem;">
                    Warm zones (red/yellow) indicate image regions with the highest gradient weight contributing to the model's prediction.
                    These are AI-assisted screening results, not clinical diagnoses. Always verify with a qualified ophthalmologist.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # TAB 3: UNCERTAINTY & OOD
        with tab_uncert:
            st.markdown("### 🧠 Monte Carlo Dropout & Out-of-Distribution Scoring")
            
            u_col1, u_col2 = st.columns([1, 1.2])

            with u_col1:
                gauge_fig = render_uncertainty_gauge(uncert['entropy'], uncert['epistemic'])
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                entropy_val = uncert.get("entropy", 0.0)
                if entropy_val < 0.6:
                    u_status_str = "Low uncertainty"
                    u_status_clr = "#34D399"
                elif entropy_val < SCREENING.uncertainty_refer_threshold:
                    u_status_str = "Moderate uncertainty"
                    u_status_clr = "#FCD34D"
                else:
                    u_status_str = "High uncertainty"
                    u_status_clr = "#F87171"

                st.markdown(f"""
                <div style="text-align: center; margin-top: -0.5rem; margin-bottom: 0.8rem;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #38BDF8;">{entropy_val:.3f} nats</span>
                    <div style="font-size: 0.82rem; font-weight: 600; color: {u_status_clr}; margin-top: 2px;">{u_status_str}</div>
                </div>
                """, unsafe_allow_html=True)

            with u_col2:
                ep_val = uncert.get("epistemic", 0.0)
                al_val = uncert.get("aleatoric", 0.0)
                ep_clr = "#34D399" if ep_val < 0.1 else ("#FCD34D" if ep_val <= 0.3 else "#F87171")
                al_clr = "#34D399" if al_val < 0.2 else ("#FCD34D" if al_val <= 0.4 else "#F87171")

                ood_val = ood.get("energy_score", 0.0)
                if ood_val <= SCREENING.ood_energy_threshold:
                    ood_chip = f'<span style="background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 2px 8px; border-radius: 6px; font-weight: 700; font-family: \'JetBrains Mono\', monospace;">{ood_val:.2f} ✓ In-distribution (threshold {SCREENING.ood_energy_threshold:.2f})</span>'
                else:
                    ood_chip = f'<span style="background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 8px; border-radius: 6px; font-weight: 700; font-family: \'JetBrains Mono\', monospace;">{ood_val:.2f} ⚠️ Out-of-distribution (threshold {SCREENING.ood_energy_threshold:.2f})</span>'

                st.markdown("#### 📐 Uncertainty Decomposition")
                st.markdown(f"""
                <div class="info-card">
                    <div style="margin-bottom: 0.8rem;">
                        <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                            <span style="color:#94A3B8;">Epistemic Uncertainty (Model Knowledge Gap):</span>
                            <b style="font-family:'JetBrains Mono'; color:{ep_clr};">{ep_val:.4f}</b>
                        </div>
                        <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
                            Indicates if this retinal pathology pattern is novel or unfamiliar to the training distribution.
                        </div>
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                            <span style="color:#94A3B8;">Aleatoric Uncertainty (Inherent Image Noise):</span>
                            <b style="font-family:'JetBrains Mono'; color:{al_clr};">{al_val:.4f}</b>
                        </div>
                        <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
                            Measures ambiguity inherent to lighting, fundus glare, or borderline lesion transitions.
                        </div>
                    </div>
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; margin-bottom: 4px;">
                            <span style="color:#94A3B8;">OOD Energy Score:</span>
                            {ood_chip}
                        </div>
                        <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
                            Threshold: &lt; {SCREENING.ood_energy_threshold:.2f} (In-distribution). Higher scores indicate non-fundus or corrupt inputs.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4: QUALITY GATE & OPTICS
        with tab_qual:
            st.markdown("### 🔍 Pre-Inference Quality Assessment")
            
            q_col1, q_col2 = st.columns([1, 1])

            with q_col1:
                st.markdown(f"""
                <div class="info-card">
                    <h4>Optic & Exposure Metrics</h4>
                    <ul style="list-style: none; padding: 0; font-size: 0.9rem; line-height: 2;">
                        <li>🔎 <b>Laplacian Blur Variance:</b> <span style="font-family: 'JetBrains Mono'; color: #38BDF8;">{qual['blur_score']:.1f}</span> (min {QUALITY.min_blur_score:.1f})</li>
                        <li>☀️ <b>Mean Luminance:</b> <span style="font-family: 'JetBrains Mono'; color: #38BDF8;">{qual['brightness']:.1f}</span> (min {QUALITY.min_brightness:.1f}, max {QUALITY.max_brightness:.1f})</li>
                        <li>🌓 <b>Pixel Standard Dev (Contrast):</b> <span style="font-family: 'JetBrains Mono'; color: #38BDF8;">{qual['contrast']:.1f}</span> (min {QUALITY.min_contrast:.1f})</li>
                        <li>📐 <b>Acquisition Dimensions:</b> <span style="font-family: 'JetBrains Mono'; color: #38BDF8;">{qual['resolution'][1]} × {qual['resolution'][0]} px</span></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with q_col2:
                if qual["passed"]:
                    st.success("✅ Quality Passed: Image meets clarity, exposure, and field-of-view criteria for automated DR screening.")
                else:
                    st.error("❌ Quality Warning: Image flagged by pre-inference validation filters.")
                    for r in qual["reasons"]:
                        st.warning(f"• {r}")

        # TAB 5: CLINICAL REPORT SUMMARY
        with tab_summary:
            st.markdown("### 📄 Formatted Clinical Screening Record")
            
            # Interactive patient info summary header
            stage_colors = ["#10B981", "#34D399", "#F59E0B", "#F97316", "#EF4444"]
            st.markdown(f"""
            <div class="info-card" style="border-left: 4px solid {stage_colors[pred_idx]}; font-family: 'Plus Jakarta Sans', sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.8rem; margin-bottom: 0.8rem;">
                    <div>
                        <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.05em;">Patient Case</span>
                        <h3 style="margin: 0.2rem 0 0 0; font-weight: 800; font-size: 1.15rem; color: #F8FAFC;">{sample_name if sample_name else 'Uploaded Fundus Scan'}</h3>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.05em;">Screening Date</span>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #F8FAFC; margin-top: 0.2rem;">{time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; font-size: 0.85rem;">
                    <div>
                        <div style="color: #64748B; font-weight: 600;">AI-assisted screening result:</div>
                        <div style="color: {stage_colors[pred_idx]}; font-weight: 700; margin-top: 2px;">Grade {pred_idx} — {pred_name}</div>
                    </div>
                    <div>
                        <div style="color: #64748B; font-weight: 600;">Confidence:</div>
                        <div style="color: #F8FAFC; font-weight: 700; margin-top: 2px; font-family: 'JetBrains Mono', monospace;">{conf * 100:.1f}%</div>
                    </div>
                    <div>
                        <div style="color: #64748B; font-weight: 600;">Referable Status:</div>
                        <div style="color: {'#EF4444' if pred_idx in SCREENING.referable_grades else '#10B981'}; font-weight: 700; margin-top: 2px;">
                            {'REFERABLE (Stage ≥ 2)' if pred_idx in SCREENING.referable_grades else 'NON-REFERABLE'}
                        </div>
                    </div>
                    <div>
                        <div style="color: #64748B; font-weight: 600;">System Quality Gate:</div>
                        <div style="color: {'#34D399' if qual['passed'] else '#EF4444'}; font-weight: 700; margin-top: 2px;">
                            {'✓ Passed' if qual['passed'] else '✗ Rejected'}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            summary_text = f"""
================================================================================
                    RETINAAI • CLINICAL SCREENING SUMMARY
================================================================================
PATIENT SCAN: {sample_name if sample_name else 'Uploaded Fundus Image'}
ANALYSIS DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}
MODEL BACKBONE: {MODEL.backbone}
--------------------------------------------------------------------------------
FINDINGS & CLASSIFICATION:
  • Predicted Stage: Grade {pred_idx} — {pred_name}
  • Confidence Score: {conf * 100:.1f}%
  • Referable Status: {'REFERABLE (Stage ≥ 2)' if pred_idx in SCREENING.referable_grades else 'NON-REFERABLE'}
  • Epistemic Uncertainty: {uncert['epistemic']:.4f} nats
  • Quality Verification: {'PASSED' if qual['passed'] else 'FAILED'}
--------------------------------------------------------------------------------
TRIAGE RECOMMENDATION:
  {decision}
--------------------------------------------------------------------------------
DISCLAIMER: AI-generated screening assistance tool for clinical research.
            Final diagnosis requires confirmation by a licensed ophthalmologist.
================================================================================
"""
            st.code(summary_text, language="text")
            st.download_button(
                "📥 Download Clinical Summary (.txt)",
                data=summary_text,
                file_name=f"RetinaAI_Report_Grade{pred_idx}_{int(time.time())}.txt",
                mime="text/plain"
            )

else:
    # ── Empty State Hero Dashboard ──────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1.5rem; background: linear-gradient(135deg, rgba(20, 32, 60, 0.4) 0%, rgba(10, 16, 32, 0.6) 100%); border-radius: 20px; border: 1px solid rgba(56, 189, 248, 0.1); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4); margin-bottom: 2.5rem;">
        <div style="font-size: 4.5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.4)); animation: pulse 3s infinite;">👁️</div>
        <h2 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 0.75rem; background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">
            Retinal Fundus Image Analysis
        </h2>
        <p style="color: #94A3B8; max-width: 700px; margin: 0 auto; font-size: 1rem; line-height: 1.6;">
            Upload a patient's fundus photograph or choose from preloaded clinical cases in the sidebar to perform automated 5-stage Diabetic Retinopathy grading, Grad-CAM++ localization, and uncertainty validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="info-card" style="height: 100%; border-top: 3px solid #38BDF8;">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔍</div>
            <h4 style="margin: 0.5rem 0; color: #38BDF8; font-weight: 800;">1. Quality Gate</h4>
            <p style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">Automated blur, under/over-exposure, and resolution validation before inference.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card" style="height: 100%; border-top: 3px solid #818CF8;">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">⚙️</div>
            <h4 style="margin: 0.5rem 0; color: #818CF8; font-weight: 800;">2. EfficientNetV2</h4>
            <p style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">EfficientNetV2-S backbone trained on APTOS 2019 for 5-tier DR severity staging.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card" style="height: 100%; border-top: 3px solid #34D399;">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">👁️</div>
            <h4 style="margin: 0.5rem 0; color: #34D399; font-weight: 800;">3. Grad-CAM++</h4>
            <p style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">Grad-CAM++ attention maps highlighting image regions contributing to the model prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="info-card" style="height: 100%; border-top: 3px solid #F59E0B;">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🧠</div>
            <h4 style="margin: 0.5rem 0; color: #F59E0B; font-weight: 800;">4. MC Dropout</h4>
            <p style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">Bayesian epistemic and aleatoric uncertainty quantification to guard against hallucinations.</p>
        </div>
        """, unsafe_allow_html=True)
