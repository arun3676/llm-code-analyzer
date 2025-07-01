#!/usr/bin/env python3
"""
Simple deployment test script to verify the Flask app works correctly.
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_app_import():
    """Test that the Flask app can be imported."""
    try:
        # Add the project root to the path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Import the app
        from code_analyzer.web.app import app
        print("✅ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Flask app: {e}")
        return False

def test_app_creation():
    """Test that the Flask app can be created."""
    try:
        from code_analyzer.web.app import app
        
        # Test basic app properties
        assert hasattr(app, 'url_map'), "App should have url_map"
        print("✅ Flask app created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create Flask app: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint."""
    try:
        from code_analyzer.web.app import app
        
        with app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200, f"Health endpoint returned {response.status_code}"
            data = response.get_json()
            assert 'status' in data, "Health response should contain status"
            print("✅ Health endpoint working")
            return True
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_main_page():
    """Test the main page loads."""
    try:
        from code_analyzer.web.app import app
        
        with app.test_client() as client:
            response = client.get('/')
            assert response.status_code == 200, f"Main page returned {response.status_code}"
            print("✅ Main page loads successfully")
            return True
    except Exception as e:
        print(f"❌ Main page test failed: {e}")
        return False

def test_environment_variables():
    """Test that environment variables are properly configured."""
    required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY', 'MERCURY_API_KEY']
    
    print("\n🔍 Checking environment variables:")
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"⚠️  {var}: Not set (will need to configure in Render)")

def main():
    """Run all deployment tests."""
    print("🚀 LLM Code Analyzer - Deployment Test")
    print("=" * 50)
    
    tests = [
        test_app_import,
        test_app_creation,
        test_health_endpoint,
        test_main_page
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    test_environment_variables()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! App is ready for deployment.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 