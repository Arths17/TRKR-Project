#!/bin/bash
# Project Reorganization Script for F1 Prediction Tracker

echo "📁 F1 Prediction Tracker - File Organization"
echo "============================================"
echo ""

# Create new directory structure
echo "Creating organized directory structure..."

# Documentation
mkdir -p docs
mkdir -p docs/guides

# Scripts
mkdir -p scripts
mkdir -p scripts/deployment
mkdir -p scripts/testing

# Source code (core ML engine)
mkdir -p src
mkdir -p src/ml
mkdir -p src/data
mkdir -p src/utils

# Frontend applications
mkdir -p apps
mkdir -p apps/streamlit

# Configuration
mkdir -p config

# Tests
mkdir -p tests

echo "✅ Directory structure created"
echo ""

# Move files
echo "Organizing files..."

# Documentation files
echo "📚 Moving documentation..."
mv -n DEPLOYMENT_GUIDE.md docs/ 2>/dev/null
mv -n F1_TRACKER_GUIDE.md docs/ 2>/dev/null
mv -n PRODUCTION_GUIDE.md docs/guides/ 2>/dev/null
mv -n QUICK_REFERENCE.md docs/ 2>/dev/null
mv -n COLUMN_FIX_SUMMARY.md docs/guides/ 2>/dev/null
mv -n FIX_SUMMARY.md docs/guides/ 2>/dev/null
mv -n IMPROVEMENTS_SUMMARY.md docs/guides/ 2>/dev/null
mv -n SYSTEM_IMPROVEMENTS.md docs/guides/ 2>/dev/null

# Scripts
echo "🔧 Moving scripts..."
mv -n deploy_github.sh scripts/deployment/ 2>/dev/null
mv -n validate_model.py scripts/testing/ 2>/dev/null
mv -n test_robust_processing.py scripts/testing/ 2>/dev/null
mv -n show_fix_details.py scripts/testing/ 2>/dev/null
mv -n GUIDE.py scripts/ 2>/dev/null

# ML and data processing
echo "🤖 Moving ML components..."
mv -n data_fetcher.py src/data/ 2>/dev/null
mv -n data_processor.py src/data/ 2>/dev/null
mv -n predictor.py src/ml/ 2>/dev/null
mv -n model_trainer.py src/ml/ 2>/dev/null
mv -n display.py src/utils/ 2>/dev/null

# Streamlit apps
echo "🎨 Moving Streamlit apps..."
mv -n f1_tracker_app.py apps/streamlit/ 2>/dev/null
mv -n streamlit_app.py apps/streamlit/ 2>/dev/null

# Configuration
echo "⚙️ Moving configuration..."
mv -n config.py config/ 2>/dev/null

# Keep in root (these are fine where they are)
# - main.py (main entry point)
# - predict_2025.py (convenience script)
# - requirements.txt
# - Dockerfile
# - README.md
# - LICENSE
# - .env.example
# - .gitignore

# Create __init__.py files for Python packages
echo "Creating __init__.py files..."
touch src/__init__.py
touch src/ml/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py

echo ""
echo "✅ File organization complete!"
echo ""
echo "📋 New structure:"
echo "├── docs/              # All documentation"
echo "├── scripts/           # Utility scripts"
echo "├── src/               # Core source code"
echo "│   ├── ml/           # ML models & training"
echo "│   ├── data/         # Data fetching & processing"
echo "│   └── utils/        # Helper utilities"
echo "├── apps/              # Frontend applications"
echo "│   └── streamlit/    # Streamlit dashboards"
echo "├── app/               # FastAPI backend"
echo "├── models/            # Trained models"
echo "├── cache/             # FastF1 cache"
echo "└── config/            # Configuration files"
echo ""
echo "⚠️  Important: Update import paths in files that reference moved modules!"
echo ""
