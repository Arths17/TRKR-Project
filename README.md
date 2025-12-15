# 🏎️ TRKR — F1 Race Tracker & AI Predictions

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red.svg)](https://streamlit.io)
[![FastF1](https://img.shields.io/badge/FastF1-3.7+-green.svg)](https://theoehrly.github.io/Fast-F1/)
[![Plotly](https://img.shields.io/badge/Plotly-6.5+-orange.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**TRKR** is a comprehensive multipage Streamlit application for F1 race analysis, live tracking, and AI-powered predictions. Access real-time FastF1 data, interactive telemetry visualizations, historical statistics, and ML-driven race forecasts all in one place.

## ✨ Core Features

### 🏁 **Race Overview** — Live Dashboard
- Year/Race/Session selector with FastF1 integration
- Race metadata card (circuit, date, session type)
- Live leaderboards (final results or practice/quali standings)
- Lap-by-lap gap evolution chart (top 10 drivers)
- Database integration (predicted vs actual leaderboards)

### 👤 **Driver Dashboard** — Telemetry & Performance
- Driver selector with instant profile loading
- Driver info card (name, team, position, points)
- Telemetry visualization (Speed/Throttle/Brake subplots)
- Lap history table (up to 50 laps with detailed metrics)
- Teammate comparison section

### 📊 **Statistics** — Season Analytics
- Championship standings (sorted by points)
- Top 10 drivers leaderboard chart
- Prediction accuracy tracking (MAE, winner%, podium%)
- Team performance aggregation
- Historical trends

### 🤖 **AI Predictions** — ML Forecasts
- Prediction selector from database
- Confidence gauge (HIGH/MEDIUM/LOW with animation)
- Feature coverage metric
- Predicted leaderboard (20 drivers with DNF risk)
- Podium predictions (P1/P2/P3)
- Post-race accuracy metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- pip or conda
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Arths17/f1tracker.git
cd f1tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running TRKR

```bash
source .venv/bin/activate
streamlit run trkr/app.py
```

Open **http://localhost:8501** in your browser.

## 📖 How It Works

### Data Sources
- **FastF1**: Real-time F1 session data (schedules, lap times, telemetry, positions)
- **SQLAlchemy ORM**: Stores predictions, evaluation metrics, and race results
- **XGBoost**: ML engine for race predictions (existing integration)

### Page Workflow

**Race Overview:**
1. Select year → race → session
2. View live race info and leaderboard
3. Analyze gap evolution over laps
4. Compare with predicted standings

**Driver Dashboard:**
1. Select year → race → session → driver
2. View driver profile (name, team, points)
3. Analyze telemetry (speed/throttle/brake)
4. Review lap history
5. Compare with teammates

**Statistics:**
1. Select season
2. View championship standings
3. Check prediction accuracy metrics
4. Analyze team performance

**AI Predictions:**
1. Select race from database
2. View confidence gauge & feature coverage
3. Review predicted leaderboard
4. Check podium predictions
5. See accuracy metrics (post-race)

## 🏗️ Project Structure

```
f1tracker/
├── trkr/                           # TRKR Multipage Streamlit App
│   ├── app.py                      # Main launcher & router (115 lines)
│   │   ├ Streamlit configuration
│   │   ├ Database initialization
│   │   └ Sidebar navigation
│   │
│   ├── pages/                      # 4 Main Pages
│   │   ├── race_overview.py        # Live race dashboard (223 lines)
│   │   ├── driver_dashboard.py     # Driver telemetry (168 lines)
│   │   ├── statistics.py           # Season analytics (197 lines)
│   │   └── ai_predictions.py       # ML forecasts (232 lines)
│   │
│   ├── utils/                      # Modular Functions
│   │   ├── live.py                 # FastF1 wrappers (206 lines, 7 functions)
│   │   ├── visuals.py              # Plotly visualizations (256 lines, 6+ functions)
│   │   ├── metrics.py              # Calculations (196 lines, 8+ functions)
│   │   ├── __init__.py
│   │   └── (utilities)
│   │
│   ├── README.md                   # TRKR-specific documentation
│   └── assets/                     # Images and static files
│
├── app/                            # F1 Tracker Backend
│   ├── database.py                 # SQLAlchemy ORM setup
│   ├── models.py                   # Race, Prediction, Metric models
│   ├── settings.py                 # Configuration
│   ├── main.py                     # FastAPI endpoints (optional)
│   ├── __init__.py
│   └── (other modules)
│
├── cache/                          # FastF1 cached data
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── README.md                       # Project README (this file)
└── .gitignore
```

## 🔌 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | 1.52.1 |
| **F1 Data Source** | FastF1 | 3.7.0 |
| **Visualization** | Plotly | 6.5.0 |
| **Data Processing** | Pandas | 2.3.3 |
| **Database ORM** | SQLAlchemy | 2.0.45 |
| **ML Engine** | XGBoost | Latest |
| **Numerical** | NumPy | 2.3.5 |
| **Python** | 3.13+ | - |

## 📊 Database Schema

```
races
├── id (PK)
├── year
├── round_number
├── race_name
├── circuit_name
└── event_date

predictions
├── id (PK)
├── race_id (FK)
├── confidence_score
├── feature_coverage
└── created_at

prediction_entries
├── id (PK)
├── prediction_id (FK)
├── driver_code
├── driver_name
├── predicted_position
├── predicted_gap
├── uncertainty
└── confidence

evaluation_metrics
├── id (PK)
├── race_id (FK)
├── position_mae
├── time_mae
├── winner_correct
├── podium_accuracy
└── skill_score
```

## 🎯 Key Capabilities

### Data Processing
- ✅ Real-time session data loading (practice, quali, race)
- ✅ Lap-by-lap analysis with gap calculations
- ✅ Telemetry extraction (speed, throttle, brake)
- ✅ Best lap identification and comparison

### Visualizations
- ✅ Interactive gap evolution charts
- ✅ Telemetry subplots (speed/throttle/brake)
- ✅ Horizontal bar leaderboards
- ✅ Confidence gauges with animations
- ✅ Accuracy heatmaps

### Metrics & Analytics
- ✅ Position prediction MAE
- ✅ Winner accuracy tracking
- ✅ Podium prediction rates
- ✅ DNF probability estimation
- ✅ Skill scoring (0-100)

### Performance
- ✅ @st.cache_resource for expensive operations
- ✅ FastF1 data caching
- ✅ Database query optimization
- ✅ Lazy database initialization

## 🚀 Deployment

### Local Testing
```bash
cd /path/to/f1tracker
source .venv/bin/activate
streamlit run trkr/app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set main file: `trkr/app.py`
4. Deploy!

### Docker (Optional)
```bash
docker build -t trkr .
docker run -p 8501:8501 trkr
```

## 📝 Recent Updates

**v2.0 - TRKR Multipage Release**
- ✅ Rebuilt as multipage Streamlit app
- ✅ 4 dedicated pages with full features
- ✅ FastF1 live data integration
- ✅ Improved visualizations (Plotly)
- ✅ AI prediction integration
- ✅ Database connectivity
- ✅ Telemetry analysis
- ✅ Historical statistics

**v1.0 - F1 Tracker Original**
- ✅ Single-page dashboard
- ✅ XGBoost predictions
- ✅ Race results tracking

## 🧪 Testing

```bash
# Verify all Python files compile
python -m py_compile trkr/app.py trkr/utils/*.py trkr/pages/*.py

# Check imports
python -c "from trkr.utils import live, visuals, metrics; print('✅ All imports OK')"

# Test database
python -c "from app.database import SessionLocal; db = SessionLocal(); print('✅ Database connected')"

# Test FastF1
python -c "from trkr.utils.live import load_season_schedule; print(f'✅ {len(load_season_schedule(2025))} races loaded')"
```

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Additional telemetry metrics (brake points, DRS usage)
- [ ] Pit stop strategy simulation
- [ ] Multi-season comparison
- [ ] Fantasy F1 integration
- [ ] Live race lap-by-lap updates
- [ ] Mobile app version
- [ ] More visualization options

## 📞 Support

- **Documentation**: See [trkr/README.md](trkr/README.md) for detailed TRKR features
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Questions? Start a GitHub Discussion

## 📜 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **[FastF1](https://github.com/theOehrly/Fast-F1)** - Official F1 timing data API
- **[Streamlit](https://streamlit.io/)** - Interactive web framework
- **[Plotly](https://plotly.com/)** - Interactive visualizations
- **[XGBoost](https://xgboost.readthedocs.io/)** - ML prediction engine
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - Database ORM

---

## 🏁 Quick Demo

```
TRKR — Race Overview
═══════════════════════════════════════════════════════════════════════════════

🏁 Abu Dhabi Grand Prix - Race 2024
───────────────────────────────────
Year: 2024 | Round: 24 | Session: Race
Circuit: Yas Marina | Location: Abu Dhabi, UAE

Final Results:
  1. Max Verstappen (RBR)    0.0s
  2. Lando Norris (McLaren) +8.3s
  3. Charles Leclerc (Ferrari) +12.1s

Gap Evolution: [Chart showing driver gaps over 58 laps]
```

---

**Built with ❤️ for F1 fans, engineers, and data enthusiasts**

**⭐ Star this repo if you found it useful!**
