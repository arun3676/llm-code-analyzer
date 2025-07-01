#!/usr/bin/env python3
"""
Demonstration of LLM Code Analyzer with mock responses
This shows how the system would work if API keys were configured
"""

import json
import time
from dataclasses import dataclass
from typing import List

@dataclass
class MockCodeAnalysisResult:
    """Mock result to demonstrate the output structure"""
    code_quality_score: float
    potential_bugs: List[str]
    improvement_suggestions: List[str]
    documentation: str
    model_name: str
    execution_time: float

def mock_analyze_code(code: str, model: str = "gpt") -> MockCodeAnalysisResult:
    """Mock function that simulates what the real analyzer would return"""
    
    # Simulate API call delay
    time.sleep(1)
    
    # Mock responses based on the code content
    if "fibonacci" in code.lower():
        return MockCodeAnalysisResult(
            code_quality_score=75.0,
            potential_bugs=[
                "Recursive implementation may cause stack overflow for large numbers",
                "No input validation for negative numbers"
            ],
            improvement_suggestions=[
                "Add input validation for negative numbers",
                "Consider iterative implementation for better performance",
                "Add docstring explaining the function purpose",
                "Consider using memoization for better performance"
            ],
            documentation="""# Fibonacci Calculator

## Description
This function calculates the nth Fibonacci number using a recursive approach.

## Function Signature
```python
def calculate_fibonacci(n: int) -> int
```

## Parameters
- `n` (int): The position in the Fibonacci sequence (0-indexed)

## Returns
- `int`: The Fibonacci number at position n

## Usage Example
```python
result = calculate_fibonacci(10)  # Returns 55
print(f"Fibonacci of 10 is: {result}")
```

## Performance Considerations
- Time complexity: O(2^n)
- Space complexity: O(n) due to recursion stack
- For large numbers, consider iterative implementation""",
            model_name=model,
            execution_time=1.2
        )
    else:
        return MockCodeAnalysisResult(
            code_quality_score=85.0,
            potential_bugs=[
                "No error handling for edge cases"
            ],
            improvement_suggestions=[
                "Add input validation",
                "Include error handling",
                "Add comprehensive documentation"
            ],
            documentation="""# Code Analysis

## Description
This appears to be a general code snippet that would benefit from additional documentation and error handling.

## Recommendations
- Add input validation
- Include error handling for edge cases
- Add comprehensive documentation
- Consider adding unit tests""",
            model_name=model,
            execution_time=0.8
        )

def generate_report(result: MockCodeAnalysisResult) -> str:
    """Generate a formatted report from analysis results"""
    
    report = f"""
{'='*60}
CODE ANALYSIS REPORT - {result.model_name.upper()} MODEL
{'='*60}

📊 QUALITY SCORE: {result.code_quality_score}/100
⏱️  EXECUTION TIME: {result.execution_time:.2f} seconds

🐛 POTENTIAL BUGS ({len(result.potential_bugs)}):
"""
    
    for i, bug in enumerate(result.potential_bugs, 1):
        report += f"  {i}. {bug}\n"
    
    report += f"""
💡 IMPROVEMENT SUGGESTIONS ({len(result.improvement_suggestions)}):
"""
    
    for i, suggestion in enumerate(result.improvement_suggestions, 1):
        report += f"  {i}. {suggestion}\n"
    
    report += f"""
📚 GENERATED DOCUMENTATION:
{result.documentation}

{'='*60}
"""
    
    return report

def main():
    """Main demonstration function"""
    
    print("🚀 LLM Code Analyzer - Demonstration Mode")
    print("=" * 50)
    print()
    print("This demonstration shows how the system would work")
    print("with properly configured API keys.")
    print()
    
    # Sample code to analyze
    sample_codes = [
        {
            "name": "Fibonacci Function",
            "code": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci of 10 is: {result}")
"""
        },
        {
            "name": "Simple Calculator",
            "code": """
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

result = add_numbers(5, 3)
print(f"5 + 3 = {result}")
"""
        }
    ]
    
    # Analyze each sample with different models
    models = ["gpt", "claude"]
    
    for sample in sample_codes:
        print(f"📝 Analyzing: {sample['name']}")
        print(f"Code:\n{sample['code']}")
        
        for model in models:
            print(f"\n🤖 Using {model.upper()} model...")
            result = mock_analyze_code(sample['code'], model)
            report = generate_report(result)
            print(report)
        
        print("\n" + "="*80 + "\n")
    
    print("✅ Demonstration complete!")
    print("\nTo use the real system:")
    print("1. Get API keys from OpenAI and Anthropic")
    print("2. Create a .env file with your keys:")
    print("   OPENAI_API_KEY=your_actual_key")
    print("   ANTHROPIC_API_KEY=your_actual_key")
    print("3. Run: python test_analyzer.py")

if __name__ == "__main__":
    main() 