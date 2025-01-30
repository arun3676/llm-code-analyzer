from typing import Dict, List, Optional, Any
import time
from functools import wraps

from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate
import json

from .models import CodeAnalysisResult, ModelEvaluationResult
from .config import DEFAULT_CONFIG
from .prompts import CODE_ANALYSIS_PROMPT, DOCUMENTATION_PROMPT
from .utils import timer_decorator, parse_llm_response
from .evaluator import ModelEvaluator

class CodeAnalyzer:
    """Main class for analyzing code using various LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the code analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.evaluator = ModelEvaluator()
        
        # Initialize LLM clients
        self.models = {
            "gpt": ChatOpenAI(
                model_name=self.config["models"]["gpt"]["name"],
                temperature=self.config["models"]["gpt"]["temperature"]
            ),
            "claude": Anthropic(
                model=self.config["models"]["claude"]["name"],
                temperature=self.config["models"]["claude"]["temperature"]
            )
        }
        
        # Initialize prompts
        self.prompts = {
            'analysis': PromptTemplate(
                input_variables=['code'],
                template=CODE_ANALYSIS_PROMPT
            ),
            'documentation': PromptTemplate(
                input_variables=['code'],
                template=DOCUMENTATION_PROMPT
            )
        }

    @timer_decorator
    def analyze_code(self, code: str, model: str = "gpt") -> CodeAnalysisResult:
        """
        Analyze code using specified LLM model.
        
        Args:
            code: Source code to analyze
            model: Model to use ('gpt' or 'claude')
            
        Returns:
            CodeAnalysisResult object containing analysis results
        """
        if model not in self.models:
            raise ValueError(f"Unsupported model: {model}")
            
        # Get code analysis
        llm = self.models[model]
        analysis_prompt = self.prompts['analysis'].format(code=code)
        analysis_result = llm.predict(analysis_prompt)
        
        # Parse results
        parsed_result = parse_llm_response(analysis_result)
        
        # Get documentation
        doc_prompt = self.prompts['documentation'].format(code=code)
        documentation = llm.predict(doc_prompt)
        
        # Create result object
        result = CodeAnalysisResult(
            code_quality_score=parsed_result['code_quality_score'],
            potential_bugs=parsed_result['potential_bugs'],
            improvement_suggestions=parsed_result['improvement_suggestions'],
            documentation=documentation,
            model_name=model,
            execution_time=0  # Will be set by timer_decorator
        )
        
        # Add to evaluator
        self.evaluator.add_result(result)
        
        return result

    def analyze_with_all_models(self, code: str) -> Dict[str, CodeAnalysisResult]:
        """Analyze code using all available models."""
        return {
            model: self.analyze_code(code, model)
            for model in self.models
        }

    def get_model_comparison(self) -> Dict[str, ModelEvaluationResult]:
        """Get comparison of model performances."""
        return self.evaluator.compare_models()

    def generate_report(self, analysis_result: CodeAnalysisResult) -> str:
        """Generate a formatted report from analysis results."""
        report = f"""
# Code Analysis Report

## Overview
- Quality Score: {analysis_result.code_quality_score}/100
- Analysis Model: {analysis_result.model_name}
- Execution Time: {analysis_result.execution_time:.2f}s

## Potential Issues
{chr(10).join(f"- {bug}" for bug in analysis_result.potential_bugs)}

## Improvement Suggestions
{chr(10).join(f"- {suggestion}" for suggestion in analysis_result.improvement_suggestions)}

## Documentation
{analysis_result.documentation}
"""
        return report