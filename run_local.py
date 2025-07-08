#!/usr/bin/env python3
"""
Local development server for LLM Code Analyzer
Run this script to start the application locally for testing
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from code_analyzer.web.app import app

if __name__ == "__main__":
    # Set development environment
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    print("🚀 Starting LLM Code Analyzer in development mode...")
    print("📱 Access the app at: http://localhost:5000")
    print("🔍 Health check at: http://localhost:5000/health")
    print("📊 Dashboard at: http://localhost:5000/dashboard")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    ) 