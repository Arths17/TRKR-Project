"""
TRKR — F1 Race Prediction & Tracking Hub
=========================================
A comprehensive multipage Streamlit app for F1 race analysis, live tracking,
and AI-powered predictions integrated with FastF1 data.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st

# Initialize database tables (lazy load to avoid circular imports)
@st.cache_resource
def init_database():
    from app.database import Base, engine
    try:
        Base.metadata.create_all(bind=engine)
        return engine
    except Exception as e:
        st.warning(f"Database initialization: {e}")
        return None

# Initialize on first run
init_database()

# Page configuration
st.set_page_config(
    page_title="TRKR — F1 Race Tracker",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Arths17/f1tracker",
        "Report a bug": "https://github.com/Arths17/f1tracker/issues",
        "About": "**TRKR** — F1 Race Prediction & Tracking Hub"
    }
)

# Custom CSS for TRKR branding
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebarNav"] {
        color: #fff;
    }
    h1, h2, h3 {
        color: #ff6b35;
    }
    .metric {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar branding
with st.sidebar:
    st.markdown("# 🏎️ TRKR")
    st.markdown("**F1 Race Prediction & Tracking**")
    st.divider()
    st.markdown("""
    Welcome to TRKR, your all-in-one F1 dashboard for:
    - 📊 **Race Overview** — Live leaderboards & gaps
    - 🏁 **Driver Dashboard** — Telemetry & performance
    - 📈 **Statistics** — Historical standings & trends
    - 🤖 **AI Predictions** — ML-powered forecasts
    """)
    st.divider()
    
    # Mode selector
    mode = st.radio("Select View", ["📊 Race Overview", "🏁 Driver Dashboard", "📈 Statistics", "🤖 AI Predictions"])

# Router logic
if mode == "📊 Race Overview":
    from pages.race_overview import show
    show()
elif mode == "🏁 Driver Dashboard":
    from pages.driver_dashboard import show
    show()
elif mode == "📈 Statistics":
    from pages.statistics import show
    show()
elif mode == "🤖 AI Predictions":
    from pages.ai_predictions import show
    show()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>TRKR © 2025 | Powered by FastF1, Streamlit & XGBoost</p>
    <p><a href='https://github.com/Arths17/f1tracker' target='_blank'>GitHub</a> | 
       <a href='#' target='_blank'>Docs</a></p>
</div>
""", unsafe_allow_html=True)
