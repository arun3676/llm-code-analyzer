import pytest
from code_analyzer import CodeAnalyzer
from code_analyzer.models import CodeAnalysisResult

# Sample code for testing
SAMPLE_CODE = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence
"""

@pytest.fixture
def analyzer():
    return CodeAnalyzer()

def test_analyze_code_with_gpt(analyzer):
    result = analyzer.analyze_code(SAMPLE_CODE, model="gpt")
    assert isinstance(result, CodeAnalysisResult)
    assert 0 <= result.code_quality_score <= 100
    assert isinstance(result.potential_bugs, list)
    assert isinstance(result.improvement_suggestions, list)
    assert isinstance(result.documentation, str)
    assert result.model_name == "gpt"
    assert result.execution_time > 0

def test_analyze_code_with_claude(analyzer):
    result = analyzer.analyze_code(SAMPLE_CODE, model="claude")
    assert isinstance(result, CodeAnalysisResult)
    assert 0 <= result.code_quality_score <= 100
    assert isinstance(result.potential_bugs, list)
    assert isinstance(result.improvement_suggestions, list)
    assert isinstance(result.documentation, str)
    assert result.model_name == "claude"
    assert result.execution_time > 0

def test_analyze_with_all_models(analyzer):
    results = analyzer.analyze_with_all_models(SAMPLE_CODE)
    assert isinstance(results, dict)
    assert "gpt" in results
    assert "claude" in results
    assert all(isinstance(r, CodeAnalysisResult) for r in results.values())

def test_model_comparison(analyzer):
    # First add some analysis results
    analyzer.analyze_code(SAMPLE_CODE, "gpt")
    analyzer.analyze_code(SAMPLE_CODE, "claude")
    
    comparison = analyzer.get_model_comparison()
    assert isinstance(comparison, dict)
    assert "gpt" in comparison
    assert "claude" in comparison
    
    for model_eval in comparison.values():
        assert 0 <= model_eval.average_quality_score <= 100
        assert model_eval.average_execution_time > 0
        assert 0 <= model_eval.success_rate <= 1

def test_invalid_model(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze_code(SAMPLE_CODE, model="invalid_model")