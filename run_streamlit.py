#!/usr/bin/env python3
"""
Streamlit App Runner for LLM Code Analyzer
Run this script to launch the Streamlit web interface
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, "code_analyzer", "web", "app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"❌ Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting LLM Code Analyzer Streamlit App...")
    print("📱 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 