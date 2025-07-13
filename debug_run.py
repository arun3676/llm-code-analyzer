import os
import logging
from dotenv import load_dotenv
import traceback

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

# API key checks
if not os.getenv('OPENAI_API_KEY'):
    logging.warning('Missing OPENAI_API_KEY - LLM calls will fail.')
if not os.getenv('ANTHROPIC_API_KEY'):
    logging.warning('Missing ANTHROPIC_API_KEY - LLM calls will fail.')
if not os.getenv('DEEPSEEK_API_KEY'):
    logging.warning('Missing DEEPSEEK_API_KEY - LLM calls will fail.')

try:
    from code_analyzer.main import CodeAnalyzer
    from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig
    from code_analyzer.performance_analyzer import PerformanceAnalyzer
    from flask import Flask

    # Simple test
    analyzer = CodeAnalyzer()
    config = AnalysisConfig()
    advanced = AdvancedCodeAnalyzer(config)
    test_code = 'def foo(): print("hello")'
    result = analyzer.analyze_code(test_code, model='deepseek', language='python')
    logging.info(f"Basic test result: {result}")

    # Test problematic code that might cause issues
    problematic_code = 'def bad(): while True: pass'
    logging.info("Testing problematic code analysis...")
    try:
        problematic_result = analyzer.analyze_code(problematic_code, model='deepseek', language='python')
        logging.info(f"Problematic code analysis result: {problematic_result}")
        logging.info(f"Quality score: {problematic_result.code_quality_score}")
        logging.info(f"Bugs found: {problematic_result.potential_bugs}")
        logging.info(f"Suggestions: {problematic_result.improvement_suggestions}")
        logging.info(f"Documentation: {problematic_result.documentation}")
    except Exception as e:
        logging.error(f"Problematic code analysis failed: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")

    # Test empty code input validation
    logging.info("Testing empty code input validation...")
    try:
        empty_result = analyzer.analyze_code("", model='deepseek', language='python')
        logging.info(f"Empty code result: {empty_result}")
    except Exception as e:
        logging.error(f"Empty code validation failed: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")

    # Test performance analyzer error handling
    logging.info("Testing performance analyzer error handling...")
    try:
        perf_analyzer = PerformanceAnalyzer()
        
        def test_function():
            return sum(i for i in range(1000))
        
        profile_result = perf_analyzer.profile_function(test_function)
        logging.info(f"Performance profiling result: {profile_result}")
        
        # Test with a function that might fail
        def failing_function():
            raise ValueError("Test error")
        
        failing_profile_result = perf_analyzer.profile_function(failing_function)
        logging.info(f"Failing function profiling result: {failing_profile_result}")
        
    except Exception as e:
        logging.error(f"Performance analyzer test failed: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")

    # Web app
    if os.path.exists('app.py'):
        from code_analyzer.web.app import app
        app.run(debug=True, port=5000)
except Exception as e:
    logging.error(f'Run failed: {str(e)}\nTraceback: {traceback.format_exc()}') 