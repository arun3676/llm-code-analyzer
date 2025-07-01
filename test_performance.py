#!/usr/bin/env python3
"""
Test script for Performance Profiling & Optimization Suggestions
Demonstrates the new performance analysis features.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from code_analyzer.performance_analyzer import PerformanceAnalyzer
from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig

def test_performance_analysis():
    """Test the performance analysis features."""
    print("🚀 Testing Performance Profiling & Optimization Suggestions")
    print("=" * 60)
    
    # Sample code with performance issues
    sample_code = """
def inefficient_fibonacci(n):
    if n <= 1:
        return n
    return inefficient_fibonacci(n-1) + inefficient_fibonacci(n-2)

def slow_list_operations():
    result = []
    for i in range(1000):
        result.append(i)  # Inefficient append in loop
    return result

def nested_loop_example():
    matrix = [[i+j for j in range(100)] for i in range(100)]
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result.append(matrix[i][j])  # Nested loops with append
    return result

def string_concatenation_issue():
    result = ""
    for i in range(1000):
        result += str(i)  # String concatenation in loop
    return result

def inefficient_lookup():
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(1000):
        if i in items:  # Inefficient list lookup
            pass
    return True
"""
    
    print("📝 Sample code with performance issues:")
    print(sample_code)
    print("\n" + "=" * 60)
    
    # Test basic performance analyzer
    print("🔍 Testing Basic Performance Analyzer...")
    analyzer = PerformanceAnalyzer()
    
    try:
        report = analyzer.analyze_code_performance(sample_code, 'python', 'test_file.py')
        
        print(f"📊 Performance Score: {report.overall_score}/100")
        print(f"🔍 Issues Found: {len(report.issues)}")
        
        for i, issue in enumerate(report.issues, 1):
            print(f"\n{i}. {issue.issue_type.upper()} ({issue.severity})")
            print(f"   Line {issue.line_number}: {issue.description}")
            print(f"   Code: {issue.code_snippet}")
            print(f"   Impact: {issue.impact}")
            print(f"   Suggestion: {issue.suggestion}")
        
        print(f"\n📋 Recommendations:")
        for rec in report.recommendations:
            print(f"   • {rec}")
            
    except Exception as e:
        print(f"❌ Error in basic analysis: {e}")
    
    print("\n" + "=" * 60)
    
    # Test function profiling
    print("⚡ Testing Function Profiling...")
    
    def test_function(n):
        """Test function for profiling."""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    try:
        profile_result = analyzer.profile_function(test_function, 10000)
        print(f"⏱️  Execution Time: {profile_result['execution_time']:.6f}s")
        print(f"📊 Profile Stats:")
        print(profile_result['profile_stats'])
        
    except Exception as e:
        print(f"❌ Error in function profiling: {e}")
    
    print("\n" + "=" * 60)
    
    # Test benchmarking alternatives
    print("🏁 Testing Benchmarking Alternatives...")
    
    def version1(n):
        """Version 1: List comprehension"""
        return [i**2 for i in range(n)]
    
    def version2(n):
        """Version 2: Traditional loop"""
        result = []
        for i in range(n):
            result.append(i**2)
        return result
    
    def version3(n):
        """Version 3: Generator expression"""
        return list(i**2 for i in range(n))
    
    try:
        code_versions = [
            ("List Comprehension", version1),
            ("Traditional Loop", version2),
            ("Generator Expression", version3)
        ]
        
        benchmark_result = analyzer.benchmark_alternatives(code_versions, 10000)
        
        print("🏆 Benchmark Results:")
        for name, result in benchmark_result.items():
            if name not in ['fastest', 'comparison']:
                if 'error' in result:
                    print(f"   {name}: Error - {result['error']}")
                else:
                    print(f"   {name}: {result['execution_time']:.6f}s")
        
        print(f"🥇 Fastest: {benchmark_result.get('fastest', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error in benchmarking: {e}")
    
    print("\n" + "=" * 60)
    
    # Test advanced analyzer with performance analysis
    print("🤖 Testing Advanced Analyzer with Performance Analysis...")
    
    try:
        config = AnalysisConfig(
            enable_rag=True,
            enable_security=True,
            enable_performance=True,
            enable_multimodal=False,
            performance_analysis_level='comprehensive'
        )
        
        advanced_analyzer = AdvancedCodeAnalyzer(config)
        
        result = advanced_analyzer.analyze_code_advanced(
            sample_code, 'python', 'test_file.py', 'deepseek'
        )
        
        if result.performance_report:
            print(f"📊 Advanced Performance Score: {result.performance_report.overall_score}/100")
            print(f"🔍 Issues Found: {len(result.performance_report.issues)}")
            
            if result.performance_report.ai_insights:
                print(f"\n🧠 AI Insights:")
                for insight in result.performance_report.ai_insights:
                    print(f"   • {insight}")
            
            if result.performance_report.optimization_examples:
                print(f"\n💡 Optimization Examples:")
                for opt in result.performance_report.optimization_examples:
                    print(f"   • {opt.get('issue_type', 'Unknown')}: {opt.get('explanation', 'No explanation')}")
        
        print(f"⏱️  Analysis Duration: {result.analysis_duration:.2f}s")
        print(f"🔧 Features Used: {', '.join(result.features_used)}")
        
    except Exception as e:
        print(f"❌ Error in advanced analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Performance Profiling & Optimization Suggestions Test Complete!")

def test_memory_profiling():
    """Test memory profiling capabilities."""
    print("\n🧠 Testing Memory Profiling...")
    
    analyzer = PerformanceAnalyzer()
    
    def memory_intensive_function(n):
        """Function that uses a lot of memory."""
        data = []
        for i in range(n):
            data.append([i] * 1000)  # Create large nested lists
        return len(data)
    
    try:
        memory_result = analyzer.profile_memory_usage(memory_intensive_function, 1000)
        print(f"⏱️  Execution Time: {memory_result['execution_time']:.6f}s")
        print(f"💾 Memory Used: {memory_result['memory_used_mb']:.2f} MB")
        print(f"📊 Result: {memory_result['result']}")
        
    except Exception as e:
        print(f"❌ Error in memory profiling: {e}")

def test_algorithm_complexity():
    """Test algorithm complexity analysis."""
    print("\n📈 Testing Algorithm Complexity Analysis...")
    
    analyzer = PerformanceAnalyzer()
    
    # Sample algorithms with different complexities
    algorithms = {
        "O(1) - Constant": "def constant_time(n): return n + 1",
        "O(n) - Linear": "def linear_time(n): return sum(range(n))",
        "O(n²) - Quadratic": """
def quadratic_time(n):
    result = 0
    for i in range(n):
        for j in range(n):
            result += i + j
    return result
""",
        "O(log n) - Logarithmic": """
def logarithmic_time(n):
    result = 0
    while n > 1:
        result += 1
        n //= 2
    return result
"""
    }
    
    for name, code in algorithms.items():
        try:
            complexity = analyzer.analyze_algorithm_complexity(code, 'python')
            print(f"🔍 {name}:")
            print(f"   Time Complexity: {complexity['time_complexity']}")
            print(f"   Space Complexity: {complexity['space_complexity']}")
            print(f"   Complexity Factors: {complexity['complexity_factors']}")
        except Exception as e:
            print(f"❌ Error analyzing {name}: {e}")

if __name__ == "__main__":
    test_performance_analysis()
    test_memory_profiling()
    test_algorithm_complexity() 