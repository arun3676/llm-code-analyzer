#!/usr/bin/env python3
"""
Performance tests for LLM Code Analyzer
Tests performance analysis functionality with mock LLM mode
"""

import sys
import os
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code_analyzer.performance_analyzer import PerformanceAnalyzer
from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig

class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use mock mode to avoid requiring real API keys
        self.analyzer = AdvancedCodeAnalyzer(mock=True)
        self.performance_analyzer = PerformanceAnalyzer()
    
    def test_performance_analyzer_initialization(self):
        """Test that performance analyzer initializes correctly."""
        self.assertIsNotNone(self.performance_analyzer)
        print("✅ Performance analyzer initialized successfully")
    
    def test_basic_performance_analysis(self):
        """Test basic performance analysis with mock data."""
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def optimized_fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
        
        try:
            result = self.performance_analyzer.analyze_code_performance(
                test_code, 'python', 'test_fibonacci.py'
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result.overall_score, (int, float))
            self.assertIsInstance(result.issues, list)
            self.assertIsInstance(result.summary, str)
            
            print(f"✅ Performance analysis completed - Score: {result.overall_score}")
            print(f"   Issues found: {len(result.issues)}")
            print(f"   Summary: {result.summary[:100]}...")
            
        except Exception as e:
            self.fail(f"Performance analysis failed: {e}")
    
    def test_complexity_analysis(self):
        """Test complexity analysis functionality."""
        test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
        
        try:
            result = self.performance_analyzer.analyze_code_performance(
                test_code, 'python', 'test_sorting.py'
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result.complexity_analysis, dict)
            
            print(f"✅ Complexity analysis completed")
            print(f"   Complexity info: {result.complexity_analysis}")
            
        except Exception as e:
            self.fail(f"Complexity analysis failed: {e}")
    
    def test_mock_analyzer_functionality(self):
        """Test that mock analyzer works correctly."""
        test_code = "def test(): pass"
        
        try:
            # Test basic analyzer in mock mode
            result = self.analyzer.base_analyzer.analyze_code(
                test_code, model='mock', language='python'
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result.code_quality_score, (int, float))
            self.assertIsInstance(result.potential_bugs, list)
            self.assertIsInstance(result.improvement_suggestions, list)
            
            print(f"✅ Mock analyzer test completed - Score: {result.code_quality_score}")
            
        except Exception as e:
            self.fail(f"Mock analyzer test failed: {e}")
    
    def test_advanced_analysis_mock(self):
        """Test advanced analysis in mock mode."""
        test_code = """
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)

def iterative_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        try:
            result = self.analyzer.analyze_code_advanced(
                test_code, language='python', file_path='test_factorial.py', model='mock'
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result.analysis_timestamp, str)
            self.assertIsInstance(result.analysis_duration, float)
            self.assertIsInstance(result.features_used, list)
            
            print(f"✅ Advanced analysis mock test completed")
            print(f"   Duration: {result.analysis_duration:.2f}s")
            print(f"   Features used: {result.features_used}")
            
        except Exception as e:
            self.fail(f"Advanced analysis mock test failed: {e}")

def run_performance_tests():
    """Run all performance tests."""
    print("🚀 Running Performance Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceAnalyzer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All performance tests passed!")
    else:
        print("❌ Some performance tests failed!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1) 