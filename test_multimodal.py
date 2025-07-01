#!/usr/bin/env python3
"""
Test script for Multi-Modal Code Analysis functionality
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_multimodal_analyzer():
    """Test the MultiModalAnalyzer class."""
    print("🧪 Testing Multi-Modal Analyzer...")
    
    try:
        from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
        
        # Initialize analyzer
        analyzer = MultiModalAnalyzer()
        
        # Check available models
        models = analyzer.get_available_models()
        print(f"✅ Available models: {models}")
        
        if not models:
            print("⚠️  No vision models available. Please check your API keys:")
            print(f"   OpenAI API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
            print(f"   Anthropic API Key: {'✅ Set' if os.getenv('ANTHROPIC_API_KEY') else '❌ Missing'}")
        else:
            print("🎉 Multi-Modal Analyzer is ready!")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_flask_endpoints():
    """Test Flask endpoints for multi-modal analysis."""
    print("\n🌐 Testing Flask endpoints...")
    
    try:
        from code_analyzer.web.app import app
        
        with app.test_client() as client:
            # Test if endpoints exist
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main page accessible")
            else:
                print(f"❌ Main page error: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Flask test error: {e}")

if __name__ == "__main__":
    print("🚀 Multi-Modal Code Analysis Test Suite")
    print("=" * 50)
    
    test_multimodal_analyzer()
    test_flask_endpoints()
    
    print("\n" + "=" * 50)
    print("📝 Next Steps:")
    print("1. Start the Flask server: python -m code_analyzer.web.app")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Go to the 'Multi-Modal Analysis' tab")
    print("4. Upload an image and test the analysis!") 