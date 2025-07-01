#!/usr/bin/env python3
"""
Test script for Interactive Code Improvement Suggestions
Tests the new fix suggestion feature.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from code_analyzer.fix_suggestions import FixSuggestionGenerator, FixSuggestion
from code_analyzer.main import CodeAnalyzer
from code_analyzer.config import DEFAULT_CONFIG

def test_fix_suggestions():
    """Test the fix suggestion generation."""
    print("🔧 Testing Interactive Code Improvement Suggestions")
    print("=" * 60)
    
    # Sample code with various issues
    sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def process_data(data_list):
    result = []
    for i in range(len(data_list)):
        for j in range(len(data_list)):
            result.append(data_list[i] + data_list[j])
    return result

def validate_user_input(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
    return cursor.fetchall()

def build_string():
    result = ""
    for i in range(1000):
        result += str(i)
    return result

def find_item(items, target):
    for item in items:
        if item == target:
            return True
    return False
"""
    
    print("📝 Sample code with various issues:")
    print(sample_code)
    print("\n" + "=" * 60)
    
    # Test basic fix suggestion generator
    print("🔍 Testing Basic Fix Suggestion Generator...")
    generator = FixSuggestionGenerator()
    
    # Create sample issues
    issues = [
        {
            'type': 'performance',
            'description': 'Inefficient recursive Fibonacci implementation',
            'line_number': 2,
            'code_snippet': 'return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)',
            'severity': 'high'
        },
        {
            'type': 'performance',
            'description': 'Nested loops causing O(n²) complexity',
            'line_number': 8,
            'code_snippet': 'for i in range(len(data_list)):\n        for j in range(len(data_list)):',
            'severity': 'medium'
        },
        {
            'type': 'security',
            'description': 'SQL injection vulnerability',
            'line_number': 15,
            'code_snippet': 'query = "SELECT * FROM users WHERE id = " + user_input',
            'severity': 'critical'
        },
        {
            'type': 'performance',
            'description': 'Inefficient string concatenation in loop',
            'line_number': 20,
            'code_snippet': 'result += str(i)',
            'severity': 'medium'
        },
        {
            'type': 'quality',
            'description': 'Inefficient list search',
            'line_number': 25,
            'code_snippet': 'for item in items:\n        if item == target:',
            'severity': 'low'
        }
    ]
    
    try:
        suggestions = generator.generate_fix_suggestions(sample_code, issues, 'python')
        
        print(f"📊 Generated {len(suggestions)} fix suggestions")
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion.title} ({suggestion.severity.upper()})")
            print(f"   Description: {suggestion.description}")
            print(f"   Confidence: {suggestion.confidence:.1%}")
            print(f"   Can auto-apply: {suggestion.can_auto_apply}")
            print(f"   Original: {suggestion.original_code}")
            print(f"   Fixed: {suggestion.fixed_code}")
            print(f"   Explanation: {suggestion.explanation[:100]}...")
            
            if suggestion.related_links:
                print(f"   Links: {', '.join(suggestion.related_links)}")
        
        # Test fix application
        print(f"\n" + "=" * 60)
        print("⚡ Testing Fix Application...")
        
        if suggestions:
            first_fix = suggestions[0]
            if first_fix.can_auto_apply:
                updated_code = generator.apply_fix(sample_code, first_fix)
                print(f"Applied fix: {first_fix.title}")
                print(f"Updated code preview: {updated_code[:200]}...")
            else:
                print(f"Fix '{first_fix.title}' cannot be auto-applied")
        
        # Test summary
        print(f"\n" + "=" * 60)
        print("📋 Fix Summary...")
        summary = generator.get_fix_summary(suggestions)
        print(f"Total suggestions: {summary['total_suggestions']}")
        print(f"Auto-applicable: {summary['auto_applicable']}")
        print(f"By severity: {summary['by_severity']}")
        print(f"By type: {summary['by_type']}")
        
    except Exception as e:
        print(f"❌ Error in fix suggestion generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # Test integration with main analyzer
    print("🔗 Testing Integration with Main Analyzer...")
    
    try:
        analyzer = CodeAnalyzer(config=DEFAULT_CONFIG, enable_rag=False)
        
        # Analyze code with fix suggestions
        result = analyzer.analyze_code(sample_code, model='deepseek')
        
        print(f"Analysis completed with quality score: {result.code_quality_score}")
        print(f"Bugs found: {len(result.potential_bugs)}")
        print(f"Suggestions: {len(result.improvement_suggestions)}")
        
        if hasattr(result, 'fix_suggestions') and result.fix_suggestions:
            print(f"Fix suggestions generated: {len(result.fix_suggestions)}")
            for i, fix in enumerate(result.fix_suggestions, 1):
                print(f"  {i}. {fix.title} ({fix.severity})")
        else:
            print("No fix suggestions generated (may need LLM client)")
        
    except Exception as e:
        print(f"❌ Error in main analyzer integration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Interactive Code Improvement Suggestions Test Complete!")

if __name__ == "__main__":
    test_fix_suggestions() 