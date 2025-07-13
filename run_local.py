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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

from code_analyzer.web.app import app

if __name__ == "__main__":
    # Set development environment
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Check for required API keys
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY not found!")
        print("Run 'python setup_api_keys.py' to configure your API keys")
    else:
        print(f"✅ Gemini API Key configured: {gemini_key[:5]}...{gemini_key[-4:]}")
    
    print("\n🚀 Starting LLM Code Analyzer in development mode...")
    print("📱 Access the app at: http://localhost:5000")
    print("🔍 Health check at: http://localhost:5000/health")
    print("📊 Dashboard at: http://localhost:5000/dashboard")
    print("🖼️  Multi-Modal Analysis tab for image analysis")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    ) 