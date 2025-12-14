"""
Streamlit dashboard launch point.
Run with: streamlit run dashboard/run.py
"""
import subprocess
import sys

if __name__ == "__main__":
    # Run the dashboard module directly
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/dashboard.py"])
