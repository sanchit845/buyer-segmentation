"""
Buyer Segmentation and Investment Profiling — Parcl Co.
========================================================
Entry point. Run this file to launch the Streamlit dashboard.

Usage:
    python main.py

Or launch directly with:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
import os


def check_dependencies():
    """Verify all required packages are installed."""
    required = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "sklearn", "scipy", "streamlit", "plotly", "joblib",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    print("[OK] All dependencies found.")


def check_data_files():
    """Verify required data and model files exist."""
    required_files = [
        "data/raw/clients.csv",
        "data/raw/properties.csv",
        "data/processed/final_segmented_dataset.csv",
        "models/clustering_model.pkl",
        "models/scaler.pkl",
        "models/cluster_mapping.pkl",
        "models/encoders.pkl",
        "models/feature_columns.pkl",
    ]
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print("[WARNING] The following files are missing:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run the notebooks in order:")
        print("  1. notebooks/01_eda.ipynb")
        print("  2. notebooks/02_preprocessing.ipynb")
        print("  3. notebooks/03_clustering.ipynb")
        print("  4. notebooks/04_business_insights.ipynb")
        sys.exit(1)
    print("[OK] All data and model files found.")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = os.path.join("dashboard", "app.py")
    if not os.path.exists(dashboard_path):
        print(f"[ERROR] Dashboard not found at: {dashboard_path}")
        sys.exit(1)

    print("\n Starting Real Estate Buyer Intelligence Dashboard...")
    print(" Open your browser at: http://localhost:8501\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        check=True,
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  Buyer Segmentation & Investment Profiling — Parcl")
    print("=" * 55)
    check_dependencies()
    check_data_files()
    launch_dashboard()