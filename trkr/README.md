# 🏎️ TRKR — F1 Race Prediction & Tracking Hub

A comprehensive, production-grade multipage Streamlit application for F1 race analysis, live tracking, and AI-powered predictions. Built with FastF1, XGBoost ML, and Plotly visualizations.

## Features

### 📊 Race Overview
- Live race dashboards and leaderboards
- Lap-by-lap gap analysis
- Real-time session tracking (FP1-FP3, Qualifying, Race)
- Integration with AI predictions
- Race information and metadata

### 🏁 Driver Dashboard
- Individual driver telemetry (speed, throttle, brake)
- Best lap analysis
- Lap history and progression
- Teammate comparisons
- Performance metrics per driver

### 📈 Statistics
- Championship standings (points, wins, podiums)
- Prediction accuracy tracking across races
- Team performance analysis
- Season-long trends
- Historical comparisons

### 🤖 AI Predictions
- ML-powered race winner/podium forecasts
- Confidence scoring (HIGH/MEDIUM/LOW)
- Feature coverage metrics
- DNF risk assessment
- Accuracy validation against actual results

## Project Structure

```
trkr/
├── app.py                          # Main Streamlit launcher
├── pages/
│   ├── race_overview.py            # Live race dashboard
│   ├── driver_dashboard.py         # Driver telemetry & performance
│   ├── statistics.py               # Historical standings & trends
│   └── ai_predictions.py           # ML prediction panel
├── utils/
│   ├── visuals.py                  # Plotly chart wrappers
│   ├── metrics.py                  # Calculation functions
│   └── live.py                     # FastF1 data fetching
├── assets/                         # Images, CSS, etc.
└── README.md                       # This file
```

## Installation & Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arths17/f1tracker.git
   cd f1predict
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run TRKR:**
   ```bash
   streamlit run trkr/app.py
   ```

   The app will open at `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add TRKR multipage app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repo
   - Set main file to: `trkr/app.py`
   - Deploy!

## Usage

### Race Overview
1. Select a **Season** (2024, 2025, etc.)
2. Choose a **Race**
3. Pick a **Session** (FP1-FP3, Qualifying, Race)
4. View:
   - Race information and circuit details
   - Live leaderboard
   - Lap-by-lap gap analysis
   - AI predictions (if available)

### Driver Dashboard
1. Select Season → Race → Session
2. Choose a **Driver** from the dropdown
3. View:
   - Driver profile and team info
   - Telemetry chart (speed, throttle, brake)
   - Lap history
   - Teammate comparisons

### Statistics
1. Select a **Season**
2. View:
   - Championship standings table
   - Top 10 drivers by points chart
   - Prediction accuracy trends
   - Team performance analysis

### AI Predictions
1. Select a **Race** with predictions
2. View:
   - Confidence gauge and feature coverage
   - Predicted leaderboard with DNF risk
   - Podium predictions
   - Accuracy metrics (if race completed)

## Data Sources

- **FastF1** — Official F1 telemetry and session data
- **XGBoost ML** — Internal prediction engine
- **SQLAlchemy + SQLite** — Prediction storage and caching

## Configuration

### Environment Variables (Optional)

Create a `.env` file:
```
DATABASE_URL=sqlite:///f1prod.db
FASTF1_CACHE=./cache/
DEBUG=false
```

### Database

The app uses SQLAlchemy ORM with SQLite by default. To use PostgreSQL:
```
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/f1tracker
```

## API Integration

TRKR integrates with the existing F1 Prediction backend:

- **Predictions** — Stored in `prediction_entries` table
- **Metrics** — Evaluation metrics in `evaluation_metrics` table
- **Race Data** — Synced from FastF1 and stored in `races` table

To generate new predictions:
```bash
python main.py --mode predict --year 2024 --race "Abu Dhabi"
```

To sync race results:
```bash
curl -X POST "http://localhost:8000/results/sync/2024/Abu\ Dhabi"
```

## Performance & Caching

- **FastF1 Cache:** Enabled globally to avoid re-fetching session data
- **Streamlit Cache:** Used for database queries and expensive computations
- **Telemetry:** Cached after first load (refresh via sidebar)

## Known Limitations

1. **Live Data Delays** — FastF1 updates may lag official timing
2. **Historical Data** — Telemetry only available for completed sessions
3. **Predictions** — Limited to races where predictions were generated
4. **Telemetry Detail** — Not available for all sessions (e.g., Practice sessions limited)

## Troubleshooting

### "No races found in database"
- Ensure predictions have been generated: `python main.py --mode predict --year 2024 --race "Qatar"`
- Sync results: `curl -X POST "http://localhost:8000/results/sync/2024/Qatar"`

### "Telemetry not available"
- Telemetry is only loaded for completed sessions
- Some FP sessions may not have detailed telemetry

### "Session not found"
- The race/session may not have occurred yet
- Check F1 calendar for scheduled dates

## Development

### Adding a New Page
1. Create `pages/new_page.py`:
   ```python
   def show():
       st.title("New Page")
       st.markdown("Content here...")
   
   if __name__ == "__main__":
       show()
   ```

2. Update `app.py` to include:
   ```python
   elif mode == "📄 New Page":
       from pages.new_page import show
       show()
   ```

### Adding Visualizations
Use wrappers in `utils/visuals.py`:
```python
from trkr.utils import visuals

fig = visuals.position_gap_chart(lap_data, "Race Name")
st.plotly_chart(fig)
```

### Adding Metrics
Use functions in `utils/metrics.py`:
```python
from trkr.utils import metrics

mae = metrics.calculate_position_mae(predictions, actuals)
skill = metrics.calculate_skill_metric(pred_df, actual_df)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Links

- **GitHub:** [Arths17/f1tracker](https://github.com/Arths17/f1tracker)
- **Streamlit Cloud:** [TRKR App](https://f1tracker.streamlit.app)
- **FastF1:** [Formula1.py](https://github.com/theOehrly/Fast-F1)
- **XGBoost:** [XGBoost Docs](https://xgboost.readthedocs.io)

---

**TRKR © 2025** | Powered by FastF1, Streamlit & XGBoost
