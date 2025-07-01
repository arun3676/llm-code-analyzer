#!/usr/bin/env python3
"""
Test script for the LLM Code Analyzer
"""

import os
from dotenv import load_dotenv
from code_analyzer.main import CodeAnalyzer

def main():
    # Load environment variables
    load_dotenv()
    
    print("=== LLM Code Analyzer Test ===")
    print()
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print("API Key Status:")
    print(f"OpenAI API Key: {'✅ SET' if openai_key else '❌ NOT SET'}")
    print(f"Anthropic API Key: {'✅ SET' if anthropic_key else '❌ NOT SET'}")
    print()
    
    # Initialize analyzer
    print("Initializing Code Analyzer...")
    try:
        analyzer = CodeAnalyzer()
        print(f"Available models: {list(analyzer.models.keys())}")
        print()
        
        if not analyzer.models:
            print("⚠️  No models available. Please set your API keys in a .env file:")
            print("OPENAI_API_KEY=your_openai_api_key")
            print("ANTHROPIC_API_KEY=your_anthropic_api_key")
            return
        
        # Test with sample code
        sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci of 10 is: {result}")
"""
        
        print("Testing with sample code:")
        print(sample_code)
        print()
        
        # Analyze with available models
        for model_name in analyzer.models.keys():
            print(f"Analyzing with {model_name.upper()} model...")
            try:
                result = analyzer.analyze_code(sample_code, model=model_name)
                print(f"✅ {model_name.upper()} Analysis Complete!")
                print(f"   Quality Score: {result.code_quality_score}/100")
                print(f"   Execution Time: {result.execution_time:.2f}s")
                print(f"   Potential Bugs: {len(result.potential_bugs)}")
                print(f"   Suggestions: {len(result.improvement_suggestions)}")
                print()
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                print()
        
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 