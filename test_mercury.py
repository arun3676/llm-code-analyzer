#!/usr/bin/env python3
"""
Test script for Mercury API integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mercury_integration():
    """Test the Mercury API integration."""
    print("🧪 Testing Mercury API Integration...")
    print("=" * 50)
    
    # Check if API key is available
    mercury_key = os.getenv("MERCURY_API_KEY")
    if not mercury_key:
        print("❌ MERCURY_API_KEY not found in environment variables")
        print("   Please add your Mercury API key to the .env file")
        return False
    
    print(f"✅ Found Mercury API key: {mercury_key[:5]}...{mercury_key[-4:]}")
    
    try:
        from code_analyzer.main import CodeAnalyzer
        
        # Initialize analyzer
        analyzer = CodeAnalyzer()
        print(f"✅ Analyzer initialized successfully")
        print(f"📊 Available models: {list(analyzer.models.keys())}")
        
        # Check if Mercury is available
        if 'mercury' not in analyzer.models:
            print("❌ Mercury model not available in analyzer")
            return False
        
        print("✅ Mercury model is available!")
        
        # Test with sample code
        sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
        
        print("\n🔍 Testing code analysis with Mercury...")
        result = analyzer.analyze_code(sample_code, model='mercury')
        
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Quality Score: {result.code_quality_score}")
        print(f"🐛 Potential Bugs: {len(result.potential_bugs)}")
        print(f"💡 Suggestions: {len(result.improvement_suggestions)}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_mercury_direct():
    """Test Mercury API directly."""
    print("\n🔧 Testing Mercury API directly...")
    print("=" * 50)
    
    try:
        from code_analyzer.main import MercuryWrapper
        
        mercury_key = os.getenv("MERCURY_API_KEY")
        if not mercury_key:
            print("❌ MERCURY_API_KEY not found")
            return False
        
        # Initialize Mercury wrapper
        mercury = MercuryWrapper(api_key=mercury_key)
        print("✅ Mercury wrapper initialized")
        
        # Test simple prompt
        test_prompt = "Hello! Can you help me with coding?"
        print(f"📤 Sending test prompt: {test_prompt}")
        
        response = mercury.invoke(test_prompt)
        print(f"📥 Received response: {response.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Mercury API Integration Test")
    print("=" * 50)
    
    # Test direct API
    direct_success = test_mercury_direct()
    
    # Test integration
    integration_success = test_mercury_integration()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    print(f"Direct API Test: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    if direct_success and integration_success:
        print("\n🎉 All tests passed! Mercury API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check your configuration.") 