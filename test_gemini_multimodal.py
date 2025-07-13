#!/usr/bin/env python3
"""
Test script for Gemini Multi-Modal Analysis functionality
This script tests the complete multimodal analysis pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_api_key():
    """Test if Gemini API key is properly configured."""
    print("🔑 Testing Gemini API Key...")
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment variables!")
        print("Please set up your API key using: python setup_api_keys.py")
        return False
    
    print(f"✅ Gemini API Key found: {gemini_key[:5]}...{gemini_key[-4:]}")
    
    # Test API connection
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash?key={gemini_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ Gemini API connection successful!")
            return True
        else:
            print(f"❌ Gemini API test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Gemini API test error: {e}")
        return False

def test_multimodal_analyzer():
    """Test the MultiModalAnalyzer class."""
    print("\n🧪 Testing MultiModalAnalyzer...")
    
    try:
        from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
        
        # Initialize analyzer
        analyzer = MultiModalAnalyzer()
        
        # Check available models
        models = analyzer.get_available_models()
        print(f"✅ Available models: {models}")
        
        if 'gemini-vision' not in models:
            print("❌ Gemini Vision not available!")
            return False
        
        print("✅ MultiModalAnalyzer initialized successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_image_preprocessing():
    """Test image preprocessing functionality."""
    print("\n🖼️  Testing Image Preprocessing...")
    
    try:
        from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
        from PIL import Image
        import io
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        test_image.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()
        
        # Test preprocessing
        analyzer = MultiModalAnalyzer()
        
        # Create a mock file object
        class MockFile:
            def __init__(self, data):
                self.data = data
                self.filename = 'test.jpg'
            
            def read(self):
                return self.data
            
            def seek(self, pos):
                pass
        
        mock_file = MockFile(image_data)
        processed_data = analyzer._preprocess_image(mock_file)
        
        if processed_data:
            print("✅ Image preprocessing successful!")
            return True
        else:
            print("❌ Image preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        return False

def test_flask_endpoints():
    """Test Flask endpoints for multimodal analysis."""
    print("\n🌐 Testing Flask Endpoints...")
    
    try:
        from code_analyzer.web.app import app
        
        with app.test_client() as client:
            # Test main page
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Main page accessible")
            else:
                print(f"❌ Main page error: {response.status_code}")
                return False
            
            # Test vision models endpoint
            response = client.get('/vision_models')
            if response.status_code == 200:
                print("✅ Vision models endpoint accessible")
            else:
                print(f"❌ Vision models endpoint error: {response.status_code}")
                return False
            
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint accessible")
            else:
                print(f"❌ Health endpoint error: {response.status_code}")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Flask test error: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n📦 Testing Dependencies...")
    
    required_packages = [
        'requests',
        'PIL',
        'flask',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available!")
    return True

def create_test_image():
    """Create a test image for analysis."""
    print("\n🎨 Creating Test Image...")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a simple test image with some text
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text to simulate code
        text = """def hello_world():
    print("Hello, World!")
    return "success"
"""
        
        # Try to use a default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text
        draw.text((20, 20), text, fill='black', font=font)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Save to file for testing
        test_image_path = 'test_code_image.jpg'
        img.save(test_image_path)
        
        print(f"✅ Test image created: {test_image_path}")
        return test_image_path
        
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Gemini Multi-Modal Analysis Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Dependencies test failed. Please install missing packages.")
        return
    
    # Test API key
    if not test_gemini_api_key():
        print("\n❌ Gemini API key test failed. Please set up your API key.")
        return
    
    # Test multimodal analyzer
    if not test_multimodal_analyzer():
        print("\n❌ MultiModalAnalyzer test failed.")
        return
    
    # Test image preprocessing
    if not test_image_preprocessing():
        print("\n❌ Image preprocessing test failed.")
        return
    
    # Test Flask endpoints
    if not test_flask_endpoints():
        print("\n❌ Flask endpoints test failed.")
        return
    
    # Create test image
    test_image_path = create_test_image()
    
    print("\n" + "=" * 60)
    print("🎉 All Tests Passed!")
    print("\n📝 Next Steps:")
    print("1. Start the application: python run_local.py")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Go to the 'Multi-Modal Analysis' tab")
    print("4. Upload an image (or use the test image: test_code_image.jpg)")
    print("5. Click 'Analyze with Gemini Vision'")
    print("\n🔗 Get your Gemini API key from: https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    main() 