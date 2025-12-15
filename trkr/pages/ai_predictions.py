"""
TRKR AI Predictions Page
=========================
ML-powered race predictions, confidence metrics, and accuracy tracking.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from trkr.utils import live, visuals, metrics

# Import database and models (will work once app is initialized)
try:
    from app.database import SessionLocal
    from app import models
except ImportError:
    st.error("Database modules not available. Ensure app is properly initialized.")
    SessionLocal = None
    models = None


def show():
    """Render AI Predictions page."""
    
    st.title("🤖 AI Predictions")
    st.markdown("ML-powered race forecasts with confidence metrics and historical accuracy")
    st.divider()
    
    if SessionLocal is None:
        st.error("Database not available. Please reload the app.")
        return
    
    # Get predictions from database
    db = SessionLocal()
    try:
        predictions = db.query(models.Prediction).all()
        
        if not predictions:
            st.info("No predictions available in database yet.")
            db.close()
            return
        
        # Convert to DataFrame for easier selection
        pred_data = [
            {
                'id': p.id,
                'race': f"R{p.race.round} - {p.race.name}" if p.race and p.race.round else f"Race {p.race_id}",
                'race_id': p.race_id,
                'confidence': p.confidence_score or 0.5,
                'feature_coverage': p.feature_coverage or 0.75
            }
            for p in predictions
        ]
        
        if not pred_data:
            st.info("No predictions available.")
            db.close()
            return
        
        pred_df = pd.DataFrame(pred_data)
        
        # ========== SELECTOR ==========
        selected_race = st.selectbox(
            "Select Race Prediction",
            options=pred_df['race'].values,
            key="ai_race_select"
        )
        
        if selected_race:
            prediction = predictions[pred_df[pred_df['race'] == selected_race].index[0]]
            
            # ========== CONFIDENCE GAUGE ==========
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence = prediction.confidence_score or 0.5
                level = "HIGH" if confidence > 0.75 else "MEDIUM" if confidence > 0.5 else "LOW"
                
                fig = visuals.prediction_confidence_gauge(confidence, level)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric(
                    "Feature Coverage",
                    f"{(prediction.feature_coverage or 0.75) * 100:.0f}%",
                    delta="Target: ≥85%" if prediction.feature_coverage and prediction.feature_coverage >= 0.85 else "Below target"
                )
            
            with col3:
                st.metric(
                    "Prediction Status",
                    "Complete" if prediction.feature_coverage and prediction.feature_coverage >= 0.85 else "Partial",
                    delta=f"{len(prediction.entries)} drivers" if prediction.entries else "0 drivers"
                )
            
            st.divider()
            
            # ========== PREDICTED LEADERBOARD ==========
            st.subheader("🏁 Predicted Leaderboard")
            
            if prediction.entries:
                entries_data = []
                confidence = prediction.confidence_score or 0.5
                confidence_pct = confidence * 100 if confidence <= 1 else confidence
                for entry in prediction.entries:
                    dnf_risk = metrics.calculate_dnf_probability(
                        entry.gap or 0,
                        confidence_pct
                    ) if entry.gap else 0
                    
                    entries_data.append({
                        'Position': entry.predicted_position,
                        'Driver': entry.driver,
                        'Gap (s)': f"{entry.gap:.2f}" if entry.gap else "N/A",
                        'Team': entry.team or "Unknown",
                        'DNF Risk': f"{dnf_risk*100:.0f}%" if dnf_risk else "Low"
                    })
                
                entries_df = pd.DataFrame(entries_data)
                
                # Highlight top 3
                def highlight_podium(row):
                    if row['Position'] in [1, 2, 3]:
                        colors = {1: 'background-color: #FFD700', 2: 'background-color: #C0C0C0', 3: 'background-color: #CD7F32'}
                        return [colors.get(row['Position'], '')] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    entries_df.style.apply(highlight_podium, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Alert for extreme gaps
                extreme_gaps = entries_df[entries_df['DNF Risk'].str.contains('%', regex=False)]
                if not extreme_gaps.empty:
                    risk_values = extreme_gaps['DNF Risk'].str.rstrip('%').astype(float)
                    high_risk = extreme_gaps[risk_values > 50]
                    if not high_risk.empty:
                        st.warning(f"⚠️ {len(high_risk)} drivers with >50% DNF risk detected")
            
            st.divider()
            
            # ========== PODIUM PREDICTIONS ==========
            st.subheader("🥇 Podium Predictions")
            
            if prediction.entries and len(prediction.entries) >= 3:
                podium = sorted(prediction.entries, key=lambda x: x.predicted_position or 999)[:3]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    p1 = podium[0] if len(podium) > 0 else None
                    st.metric(
                        "🥇 P1",
                        p1.driver if p1 else "N/A",
                        f"Gap: {p1.gap:.1f}s" if p1 and p1.gap else "N/A"
                    )
                
                with col2:
                    p2 = podium[1] if len(podium) > 1 else None
                    st.metric(
                        "🥈 P2",
                        p2.driver if p2 else "N/A",
                        f"Gap: {p2.gap:.1f}s" if p2 and p2.gap else "N/A"
                    )
                
                with col3:
                    p3 = podium[2] if len(podium) > 2 else None
                    st.metric(
                        "🥉 P3",
                        p3.driver if p3 else "N/A",
                        f"Gap: {p3.gap:.1f}s" if p3 and p3.gap else "N/A"
                    )
            
            st.divider()
            
            # ========== ACCURACY METRICS (if race completed) ==========
            st.subheader("📊 Accuracy Metrics")
            
            eval_metrics = db.query(models.EvaluationMetric).filter(
                models.EvaluationMetric.race_id == prediction.race_id
            ).first()
            
            if eval_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Position MAE",
                        f"{eval_metrics.position_mae:.2f}" if eval_metrics.position_mae else "N/A",
                        delta="Lower is better"
                    )
                
                with col2:
                    winner_correct = "✅ Correct" if eval_metrics.winner_correct else "❌ Incorrect"
                    st.metric("Winner Prediction", winner_correct)
                
                with col3:
                    podium_accuracy = f"{eval_metrics.podium_accuracy*100:.0f}%" if eval_metrics.podium_accuracy else "N/A"
                    st.metric("Podium Accuracy", podium_accuracy)
                
                with col4:
                    skill = f"{eval_metrics.skill_score:.0f}/100" if eval_metrics.skill_score else "N/A"
                    st.metric("Skill Score", skill)
            else:
                st.info("Race not yet completed. Accuracy metrics will appear after the race.")
            
            # ========== DETAILED ANALYSIS ==========
            with st.expander("📈 Detailed Performance Analysis"):
                st.write("**Model Characteristics:**")
                st.write(f"- Confidence: {(prediction.confidence_score or 0.5)*100:.0f}%")
                st.write(f"- Feature Coverage: {(prediction.feature_coverage or 0.75)*100:.0f}%")
                st.write(f"- Predictions Made: {len(prediction.entries)} drivers")
                
                if eval_metrics:
                    st.write("\n**Historical Performance:**")
                    st.write(f"- Average Position Error: {eval_metrics.position_mae:.2f} positions")
                    st.write(f"- Winner Accuracy: {'Yes' if eval_metrics.winner_correct else 'No'}")
                    st.write(f"- Podium Prediction Rate: {eval_metrics.podium_accuracy*100:.0f}%")
    
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
    
    finally:
        db.close()


if __name__ == "__main__":
    show()
