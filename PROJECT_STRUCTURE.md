# 📁 F1 Prediction Tracker - Project Structure

## 🎯 Organized Directory Layout

```
f1predict/
│
├── 📚 docs/                          # All Documentation
│   ├── DEPLOYMENT_GUIDE.md          # GitHub deployment instructions
│   ├── F1_TRACKER_GUIDE.md          # Complete user guide
│   ├── QUICK_REFERENCE.md           # Quick reference card
│   └── guides/                       # Additional guides
│       ├── PRODUCTION_GUIDE.md      # Production deployment
│       ├── IMPROVEMENTS_SUMMARY.md  # System improvements log
│       └── FIX_SUMMARY.md           # Bug fixes log
│
├── 🔧 scripts/                       # Utility Scripts
│   ├── deployment/                   # Deployment scripts
│   │   └── deploy_github.sh         # GitHub deployment automation
│   ├── testing/                      # Testing utilities
│   │   ├── validate_model.py        # Model validation
│   │   ├── test_robust_processing.py # Data processing tests
│   │   └── show_fix_details.py      # System diagnostics
│   └── GUIDE.py                      # Helper scripts guide
│
├── 🤖 src/                           # Core Source Code
│   ├── __init__.py
│   ├── ml/                           # Machine Learning
│   │   ├── __init__.py
│   │   ├── predictor.py             # Prediction engine
│   │   └── model_trainer.py         # Model training
│   ├── data/                         # Data Management
│   │   ├── __init__.py
│   │   ├── data_fetcher.py          # FastF1 data fetching
│   │   └── data_processor.py        # Data processing & cleaning
│   └── utils/                        # Helper Utilities
│       ├── __init__.py
│       └── display.py               # Display formatting
│
├── 🎨 apps/                          # Frontend Applications
│   └── streamlit/                    # Streamlit Dashboards
│       ├── f1_tracker_app.py        # Main production app ⭐
│       └── streamlit_app.py         # Alternative dashboard
│
├── ⚡ app/                           # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                      # API entrypoint
│   ├── models.py                    # SQLAlchemy ORM
│   ├── schemas.py                   # Pydantic models
│   ├── services.py                  # Business logic
│   ├── api.py                       # Route handlers
│   ├── database.py                  # DB connection
│   └── settings.py                  # Configuration
│
├── 🧠 models/                        # Trained ML Models
│   ├── FinishPosition_xgboost.pkl
│   ├── scaler_FinishPosition.pkl
│   └── feature_columns.pkl
│
├── 💾 cache/                         # FastF1 Data Cache
│   └── [fastf1 cached data]
│
├── ⚙️ config/                        # Configuration Files
│   └── config.py                     # App configuration
│
├── 🧪 tests/                         # Test Suite (future)
│   └── [test files]
│
├── 📋 Root Files                     # Core Project Files
│   ├── main.py                      # CLI entry point
│   ├── predict_2025.py              # Convenience prediction script
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Container config
│   ├── .env.example                 # Environment template
│   ├── .gitignore                   # Git exclusions
│   ├── README.md                    # Main documentation
│   └── LICENSE                      # MIT License
│
└── 🔄 Generated/Runtime              # Generated at runtime
    ├── .venv/                        # Virtual environment
    ├── __pycache__/                  # Python cache
    ├── f1prod.db                     # SQLite database
    ├── predictions/                  # Prediction outputs
    └── dashboard/                    # Legacy dashboards
```

---

## 🎯 Quick Navigation

### For Users
- **Start Here:** [README.md](../README.md)
- **User Guide:** [docs/F1_TRACKER_GUIDE.md](F1_TRACKER_GUIDE.md)
- **Quick Reference:** [docs/QUICK_REFERENCE.md](QUICK_REFERENCE.md)

### For Developers
- **Production Guide:** [docs/guides/PRODUCTION_GUIDE.md](guides/PRODUCTION_GUIDE.md)
- **API Backend:** [app/](../app/)
- **ML Engine:** [src/ml/](../src/ml/)
- **Data Pipeline:** [src/data/](../src/data/)

### For Deployment
- **GitHub Deploy:** [scripts/deployment/deploy_github.sh](../scripts/deployment/deploy_github.sh)
- **Deployment Guide:** [docs/DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Docker:** [Dockerfile](../Dockerfile)

---

## 📦 Module Organization

### `src/` - Core Library
All reusable, independent code that doesn't depend on specific frameworks.

**`src/ml/`** - Machine Learning Components
- `predictor.py`: XGBoost prediction engine
- `model_trainer.py`: Model training pipeline

**`src/data/`** - Data Management
- `data_fetcher.py`: FastF1 API integration
- `data_processor.py`: Data cleaning & feature engineering

**`src/utils/`** - Shared Utilities
- `display.py`: Console output formatting

### `apps/` - Applications
Framework-specific applications built on top of `src/`.

**`apps/streamlit/`** - Web Dashboards
- `f1_tracker_app.py`: Production dashboard (USE THIS ONE ⭐)
- `streamlit_app.py`: Alternative implementation

### `app/` - Backend API
FastAPI REST API for predictions and data management.

### `docs/` - Documentation
All user-facing and developer documentation.

### `scripts/` - Automation
Scripts for deployment, testing, and maintenance.

---

## 🔄 Import Patterns

After reorganization, use these import patterns:

### From Streamlit Apps
```python
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.predictor import predict_race
from src.data.data_fetcher import fetch_race_data
```

### From Root Scripts
```python
from src.ml.model_trainer import train_models
from src.data.data_processor import process_data
```

### From FastAPI Backend
```python
# Backend is separate, uses direct imports
from app.services import PredictionService
from app.models import Race, Prediction
```

---

## 🚀 Running Applications

### Main Streamlit App (Recommended)
```bash
streamlit run apps/streamlit/f1_tracker_app.py
```

### FastAPI Backend
```bash
uvicorn app.main:app --reload
```

### CLI Predictions
```bash
python main.py --mode predict --year 2024 --race "Abu Dhabi"
```

### Training Models
```bash
python main.py --mode train --seasons 2023 2024
```

---

## 🎨 File Categories

### 📝 Documentation (`.md`)
- User guides, API docs, deployment instructions
- Location: `docs/`

### 🐍 Source Code (`.py`)
- **Core ML:** `src/ml/`
- **Data pipeline:** `src/data/`
- **Web apps:** `apps/streamlit/`
- **API backend:** `app/`

### 🔧 Scripts (`.sh`, `.py`)
- **Deployment:** `scripts/deployment/`
- **Testing:** `scripts/testing/`

### ⚙️ Configuration
- `.env.example`: Environment template
- `config/`: App configuration
- `requirements.txt`: Dependencies
- `Dockerfile`: Container setup

### 🗃️ Data & Models
- `models/`: Trained ML models (`.pkl`)
- `cache/`: FastF1 cache
- `f1prod.db`: SQLite database
- `predictions/`: Generated predictions

---

## 🧹 Cleanup Recommendations

### Files to Keep
✅ All source code (`.py`)
✅ Documentation (`.md`)
✅ Configuration files
✅ Trained models (`models/*.pkl`)
✅ Scripts (`scripts/`)

### Files Safe to Delete
❌ `__pycache__/` - Python cache (regenerated)
❌ `*.pyc` - Compiled Python
❌ `.DS_Store` - macOS files
❌ `venv/` - Virtual environment (recreate with `pip install -r requirements.txt`)

### Files to Review
⚠️ `dashboard/` - Legacy dashboards (keep if used, otherwise archive)
⚠️ `predictions/` - Old predictions (archive if not needed)
⚠️ `f1prod.db` - Database (backup before deleting)

---

## 📊 Reorganization Checklist

- [ ] Run `./reorganize.sh` to move files
- [ ] Update import statements in moved files
- [ ] Test Streamlit app: `streamlit run apps/streamlit/f1_tracker_app.py`
- [ ] Test FastAPI backend: `uvicorn app.main:app`
- [ ] Test CLI: `python main.py --help`
- [ ] Update documentation links
- [ ] Commit changes to git
- [ ] Update `.gitignore` if needed

---

## 🔍 Find Files Quickly

```bash
# Find all Python files
find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*"

# Find documentation
find docs/ -name "*.md"

# Find scripts
find scripts/ -type f

# Find models
find models/ -name "*.pkl"
```

---

## 💡 Best Practices

### 1. Keep Root Clean
- Only essential files in root directory
- Move detailed docs to `docs/`
- Move scripts to `scripts/`

### 2. Separate Concerns
- ML code in `src/ml/`
- Data code in `src/data/`
- Web apps in `apps/`
- API in `app/`

### 3. Clear Naming
- Use descriptive folder names
- Group related files together
- Keep `__init__.py` in Python packages

### 4. Documentation
- README in root for overview
- Detailed guides in `docs/`
- Code comments for complex logic

---

## 🆘 Troubleshooting

### Import Errors After Reorganization
```python
# Add this to top of files that can't find modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Streamlit Not Finding App
```bash
# Use full path
streamlit run /full/path/to/apps/streamlit/f1_tracker_app.py

# Or navigate first
cd /Users/atharvranjan/f1predict
streamlit run apps/streamlit/f1_tracker_app.py
```

---

**Last Updated:** December 14, 2025
**Version:** 2.0.0 (Reorganized Structure)
