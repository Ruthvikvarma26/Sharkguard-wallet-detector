#!/usr/bin/env python3
"""
SharkGuard Application Launcher
================================

This script provides an easy way to start the SharkGuard web application.
It handles setup and provides helpful information.

Usage:
    python run_app.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import sklearn
        import matplotlib
        import requests
        import joblib
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the model exists, if not, generate training data."""
    model_path = Path("models/isolation_model.joblib")
    if model_path.exists():
        print("✅ Model found")
        return True
    else:
        print("⚠️  Model not found. Generating training data...")
        try:
            # Generate simulated data
            subprocess.run([sys.executable, "data/simulate.py"], check=True)
            print("✅ Training data generated")
            print("ℹ️  You can now train the model from the web interface")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to generate training data")
            return False

def main():
    """Main launcher function."""
    print("🦈 SharkGuard - Web3 Fake Account Detector")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where app.py is located)")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    check_model()
    
    print("\n🚀 Starting SharkGuard web application...")
    print("   The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 SharkGuard stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
