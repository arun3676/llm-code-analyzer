from typing import Dict, List, Optional, Any
import time
import os
from functools import wraps
from dotenv import load_dotenv
import traceback
import openai  # Make sure to import openai directly

# Load environment variables at the top of the file
load_dotenv()

# Import with error handling
try:
    from langchain.prompts import PromptTemplate
    
    # Try to import Anthropic integration
    try:
        from langchain_anthropic import ChatAnthropic
        ANTHROPIC_AVAILABLE = True
    except ImportError:
        print("Warning: langchain_anthropic not available. Falling back to langchain.chat_models")
        try:
            from langchain.chat_models import ChatAnthropic
            ANTHROPIC_AVAILABLE = True
        except ImportError:
            print("Error: Could not import ChatAnthropic from any package")
            ANTHROPIC_AVAILABLE = False
            
except ImportError as e:
    print(f"Error importing required packages: {e}")
    raise

import json

from .models import CodeAnalysisResult, ModelEvaluationResult
from .config import DEFAULT_CONFIG
from .prompts import CODE_ANALYSIS_PROMPT, DOCUMENTATION_PROMPT
from .utils import timer_decorator, parse_llm_response
from .evaluator import ModelEvaluator

# Create a simple wrapper for OpenAI
class SimpleOpenAIWrapper:
    def __init__(self, api_key, model_name="gpt-3.5-turbo", temperature=0.1):
        self.model_name = model_name
        self.temperature = temperature
        openai.api_key = api_key
        
    def invoke(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        
        class SimpleResponse:
            def __init__(self, content):
                self.content = content
        
        return SimpleResponse(response.choices[0].message.content)

# Set default model names
DEFAULT_GPT_MODEL = "gpt-3.5-turbo"
DEFAULT_CLAUDE_MODEL = "claude-3-haiku-20240307"

class CodeAnalyzer:
    """Main class for analyzing code using various LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the code analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.evaluator = ModelEvaluator()
        
        # Debug config
        print(f"Using config: {self.config}")
        
        # Check for API keys
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not openai_api_key:
            print("WARNING: OPENAI_API_KEY not found in environment variables")
        else:
            print(f"Found OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-4:]}")
        
        if not anthropic_api_key:
            print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
        else:
            print(f"Found Anthropic API key: {anthropic_api_key[:5]}...{anthropic_api_key[-4:]}")
        
        # Initialize LLM clients
        self.models = {}
        
        # Initialize OpenAI with simple wrapper
        if openai_api_key:
            try:
                # Get model details from config
                gpt_model_name = self.config.get("models", {}).get("gpt", {}).get("name", DEFAULT_GPT_MODEL)
                gpt_temp = self.config.get("models", {}).get("gpt", {}).get("temperature", 0.1)
                
                print(f"Initializing GPT model: {gpt_model_name} with temperature {gpt_temp}")
                
                # Test OpenAI connection
                test_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5
                )
                print(f"OpenAI test successful: {test_response.choices[0].message.content}")
                
                # Create simple wrapper
                self.models["gpt"] = SimpleOpenAIWrapper(
                    api_key=openai_api_key,
                    model_name=gpt_model_name,
                    temperature=gpt_temp
                )
                print("Successfully initialized GPT model")
            except Exception as e:
                print(f"Error initializing GPT model: {e}")
                traceback.print_exc()
        
        # Initialize Claude model
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                # Get model details from config
                claude_model_name = self.config.get("models", {}).get("claude", {}).get("name", DEFAULT_CLAUDE_MODEL)
                claude_temp = self.config.get("models", {}).get("claude", {}).get("temperature", 0.1)
                
                print(f"Initializing Claude model: {claude_model_name} with temperature {claude_temp}")
                
                # Use ChatAnthropic instead of Anthropic
                self.models["claude"] = ChatAnthropic(
                    model=claude_model_name,
                    temperature=claude_temp,
                    anthropic_api_key=anthropic_api_key
                )
                print("Successfully initialized Claude model")
            except Exception as e:
                print(f"Error initializing Claude model: {e}")
                traceback.print_exc()
        
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
        
        # Print available models
        print(f"Available models: {list(self.models.keys())}")

    # The rest of the class remains the same
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
            raise ValueError(f"Unsupported model: {model}. Available models: {list(self.models.keys())}")
            
        # Get code analysis
        llm = self.models[model]
        analysis_prompt = self.prompts['analysis'].format(code=code)
        
        print(f"Sending analysis prompt to {model} model...")
        try:
            # Use invoke method for all models
            result = llm.invoke(analysis_prompt)
            analysis_result = result.content if hasattr(result, 'content') else str(result)
            print(f"Received response from {model} model: {analysis_result[:100]}...")  # Print start of response
        except Exception as e:
            print(f"Error getting analysis from {model} model: {e}")
            traceback.print_exc()
            analysis_result = f"Error: {str(e)}"
        
        # Parse results
        try:
            print(f"Parsing analysis result...")
            parsed_result = parse_llm_response(analysis_result)
            print(f"Parsed result: {parsed_result}")
        except Exception as e:
            print(f"Error parsing analysis result: {e}")
            traceback.print_exc()
            parsed_result = {
                "code_quality_score": 50,
                "potential_bugs": ["Error parsing model response"],
                "improvement_suggestions": ["Try again with different parameters"],
                "documentation": "Error generating documentation"
            }
        
        # Get documentation
        doc_prompt = self.prompts['documentation'].format(code=code)
        
        print(f"Sending documentation prompt to {model} model...")
        try:
            # Use invoke method for all models
            result = llm.invoke(doc_prompt)
            documentation = result.content if hasattr(result, 'content') else str(result)
            print(f"Received documentation from {model} model: {documentation[:100]}...")  # Print start of response
        except Exception as e:
            print(f"Error getting documentation from {model} model: {e}")
            traceback.print_exc()
            documentation = f"Error generating documentation: {str(e)}"
        
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