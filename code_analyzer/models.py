from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class CodeAnalysisResult:
    """Container for code analysis results."""
    code_quality_score: float
    potential_bugs: List[str]
    improvement_suggestions: List[str]
    documentation: str
    model_name: str
    execution_time: float

@dataclass
class ModelEvaluationResult:
    """Container for model evaluation results."""
    model_name: str
    average_quality_score: float
    average_execution_time: float
    success_rate: float
    analysis_samples: List[CodeAnalysisResult]