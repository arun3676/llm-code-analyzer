from typing import Dict, List, Optional, Any
import time
import os
from functools import wraps
from dotenv import load_dotenv
import traceback
import concurrent.futures
import re
import logging
import shutil

# Load environment variables at the top of the file
load_dotenv()

# Import with error handling
try:
    from langchain.prompts import PromptTemplate
    ANTHROPIC_AVAILABLE = True  # We'll use direct Anthropic client
except ImportError as e:
    logging.warning(f'langchain missing - fallback to basic mode: {e}')
    ANTHROPIC_AVAILABLE = False

import json

from .models import CodeAnalysisResult, ModelEvaluationResult
from .config import DEFAULT_CONFIG
from .prompts import CODE_ANALYSIS_PROMPT, DOCUMENTATION_PROMPT
from .utils import timer_decorator, parse_llm_response
from .evaluator import ModelEvaluator

# Import fix suggestion generator
try:
    from .fix_suggestions import FixSuggestionGenerator
    FIX_SUGGESTIONS_AVAILABLE = True
except ImportError:
    logging.warning('fix_suggestions missing - fallback to basic mode')
    FIX_SUGGESTIONS_AVAILABLE = False

# Import language detector
try:
    from .language_detector import LanguageDetector
    LANGUAGE_DETECTOR_AVAILABLE = True
except ImportError:
    logging.warning('language_detector missing - fallback to basic mode')
    LANGUAGE_DETECTOR_AVAILABLE = False

# Import new analyzers
try:
    from .framework_analyzer import FrameworkAnalyzer
    FRAMEWORK_ANALYZER_AVAILABLE = True
except ImportError:
    logging.warning('framework_analyzer missing - fallback to basic mode')
    FRAMEWORK_ANALYZER_AVAILABLE = False

try:
    from .cloud_analyzer import CloudAnalyzer
    CLOUD_ANALYZER_AVAILABLE = True
except ImportError:
    logging.warning('cloud_analyzer missing - fallback to basic mode')
    CLOUD_ANALYZER_AVAILABLE = False

try:
    from .container_analyzer import ContainerAnalyzer
    CONTAINER_ANALYZER_AVAILABLE = True
except ImportError:
    logging.warning('container_analyzer missing - fallback to basic mode')
    CONTAINER_ANALYZER_AVAILABLE = False

# DeepSeek wrapper using OpenAI SDK
class DeepSeekWrapper:
    def __init__(self, api_key, model_name="deepseek-chat", temperature=0.1):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False
            )
            class SimpleResponse:
                def __init__(self, content):
                    self.content = content
            return SimpleResponse(response.choices[0].message.content)
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return type('ErrorResponse', (), {'content': f'Error: {str(e)}'})()

# Mercury wrapper using OpenAI SDK
class MercuryWrapper:
    def __init__(self, api_key, model_name="mercury-coder", temperature=0.1):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.inceptionlabs.ai/v1"
        )
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False
            )
            print("[MercuryWrapper] Raw response:", response)
            class SimpleResponse:
                def __init__(self, content):
                    self.content = content
            return SimpleResponse(response.choices[0].message.content)
        except Exception as e:
            print(f"Mercury API error: {e}")
            return type('ErrorResponse', (), {'content': f'Error: {str(e)}'})()

# Set default model names
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_CLAUDE_MODEL = "claude-3-haiku-20240307"
DEFAULT_MERCURY_MODEL = "mercury"

class CodeAnalyzer:
    """Main class for analyzing code using various LLM models with integrated RAG capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, mock: bool = False):
        """Initialize the code analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.evaluator = ModelEvaluator()
        
        # Slim ChromaDB
        try:
            import chromadb
            self.vector_db = chromadb.EphemeralClient()
            # Example: limit to 50 chunks when adding texts
            # self.vector_db.add_texts(texts[:50])  # Uncomment and adapt as needed
            shutil.rmtree('chroma_data', ignore_errors=True)
        except Exception as e:
            logging.warning(f'ChromaDB slimming failed: {e}')
        
        # Initialize specialized analyzers
        if FRAMEWORK_ANALYZER_AVAILABLE:
            self.framework_analyzer = FrameworkAnalyzer()
        else:
            self.framework_analyzer = None
            
        if CLOUD_ANALYZER_AVAILABLE:
            self.cloud_analyzer = CloudAnalyzer()
        else:
            self.cloud_analyzer = None
            
        if CONTAINER_ANALYZER_AVAILABLE:
            self.container_analyzer = ContainerAnalyzer()
        else:
            self.container_analyzer = None
        
        # Debug config
        print(f"Using config: {self.config}")
        
        # Check for API keys with proper validation
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        mercury_api_key = os.environ.get("MERCURY_API_KEY")
        
        if not deepseek_api_key and not mock:
            raise ValueError('Missing DEEPSEEK_API_KEY - set in .env')
        else:
            print(f"Found DeepSeek API key: {deepseek_api_key[:5]}...{deepseek_api_key[-4:] if deepseek_api_key else 'None'}")
        
        if not anthropic_api_key and not mock:
            raise ValueError('Missing ANTHROPIC_API_KEY - set in .env')
        else:
            print(f"Found Anthropic API key: {anthropic_api_key[:5]}...{anthropic_api_key[-4:] if anthropic_api_key else 'None'}")
        
        if not mercury_api_key and not mock:
            raise ValueError('Missing MERCURY_API_KEY - set in .env')
        else:
            print(f"Found Mercury API key: {mercury_api_key[:5]}...{mercury_api_key[-4:] if mercury_api_key else 'None'}")
        
        # Initialize LLM clients
        self.models = {}
        
        if mock:
            # Mock mode for testing
            self.models["mock"] = lambda prompt: type('MockResponse', (), {'content': 'Mock response: Analyzed'})()
            print("Initialized in mock mode")
            return
        
        # Register DeepSeek
        if deepseek_api_key:
            try:
                deepseek_model_name = self.config.get("models", {}).get("deepseek", {}).get("name", DEFAULT_DEEPSEEK_MODEL)
                deepseek_temp = self.config.get("models", {}).get("deepseek", {}).get("temperature", 0.1)
                print(f"Initializing DeepSeek model: {deepseek_model_name} with temperature {deepseek_temp}")
                self.models["deepseek"] = DeepSeekWrapper(
                    api_key=deepseek_api_key,
                    model_name=deepseek_model_name,
                    temperature=deepseek_temp
                )
                # Test DeepSeek connection
                try:
                    test_response = self.models["deepseek"].invoke("Test")
                    print(f"DeepSeek test successful: {test_response.content}")
                except Exception as test_exc:
                    print(f"DeepSeek test failed: {test_exc}")
                    del self.models["deepseek"]
            except Exception as e:
                print(f"Error initializing DeepSeek model: {e}")
                traceback.print_exc()
        
        # Initialize Claude model
        if anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                # Get model details from config
                claude_model_name = self.config.get("models", {}).get("claude", {}).get("name", DEFAULT_CLAUDE_MODEL)
                claude_temp = self.config.get("models", {}).get("claude", {}).get("temperature", 0.1)
                
                print(f"Initializing Claude model: {claude_model_name} with temperature {claude_temp}")
                
                # Use direct Anthropic client to avoid proxy issues
                try:
                    import anthropic
                    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                    class AnthropicWrapper:
                        def __init__(self, client, model_name, temperature):
                            self.client = client
                            self.model_name = model_name
                            self.temperature = temperature
                        def invoke(self, prompt):
                            try:
                                response = self.client.messages.create(
                                    model=self.model_name,
                                    max_tokens=2000,
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                class SimpleResponse:
                                    def __init__(self, content):
                                        self.content = content
                                return SimpleResponse(response.content[0].text)
                            except Exception as e:
                                print(f"Anthropic API error: {e}")
                                return type('ErrorResponse', (), {'content': f'Error: {str(e)}'})()
                    self.models["claude"] = AnthropicWrapper(anthropic_client, claude_model_name, claude_temp)
                    print("Successfully initialized Claude model")
                except ImportError:
                    print("Anthropic library not available, skipping Claude initialization")
            except Exception as e:
                print(f"Error initializing Claude model: {e}")
                traceback.print_exc()
        
        # Register Mercury
        if mercury_api_key:
            try:
                mercury_model_name = self.config.get("models", {}).get("mercury", {}).get("name", DEFAULT_MERCURY_MODEL)
                mercury_temp = self.config.get("models", {}).get("mercury", {}).get("temperature", 0.1)
                print(f"Initializing Mercury model: {mercury_model_name} with temperature {mercury_temp}")
                self.models["mercury"] = MercuryWrapper(
                    api_key=mercury_api_key,
                    model_name=mercury_model_name,
                    temperature=mercury_temp
                )
                # Test Mercury connection
                try:
                    test_response = self.models["mercury"].invoke("Test")
                    print(f"Mercury test successful: {test_response.content}")
                except Exception as test_exc:
                    print(f"Mercury test failed: {test_exc}")
                    del self.models["mercury"]
            except Exception as e:
                print(f"Error initializing Mercury model: {e}")
                traceback.print_exc()
        
        # Initialize fix suggestion generator
        self.fix_generator = None
        if FIX_SUGGESTIONS_AVAILABLE:
            try:
                # Use the first available model for fix generation
                llm_client = next(iter(self.models.values())) if self.models else None
                self.fix_generator = FixSuggestionGenerator(llm_client)
                print("Fix suggestion generator initialized successfully!")
            except Exception as e:
                print(f"Error initializing fix generator: {e}")
                traceback.print_exc()
        
        # Initialize language detector
        self.language_detector = LanguageDetector() if LANGUAGE_DETECTOR_AVAILABLE else None
        
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

    @timer_decorator
    def analyze_code(self, code: str, model: str = "deepseek", file_path: Optional[str] = None, language: Optional[str] = None, mode: str = "quick") -> CodeAnalysisResult:
        """
        Analyze code using specified LLM model with optional RAG suggestions and quick/thorough mode.
        
        Args:
            code: Source code to analyze
            model: Model to use ('deepseek' or 'claude')
            file_path: Path to the file containing the code
            language: Optional language of the code
            mode: Analysis mode ('quick' or 'thorough')
            
        Returns:
            CodeAnalysisResult object containing analysis results
        """
        try:
            # Input validation
            if not code or not code.strip():
                raise ValueError('Empty code input')
            
            if model not in self.models:
                raise ValueError(f"Unsupported model: {model}. Available models: {list(self.models.keys())}")
                
            # Always use override if set, else detect
            detected_language = language
            if not detected_language and hasattr(self, 'language_detector') and self.language_detector:
                try:
                    lang_info = self.language_detector.detect_language(code, file_path)
                    detected_language = lang_info.name
                except Exception as e:
                    logging.warning(f'Language detection failed: {e}')
                    detected_language = 'python'
            if not detected_language:
                detected_language = 'python'
            
            # Get code analysis
            llm = self.models[model]
            
            # Add language context to prompt
            lang_context = f"The code is written in {detected_language}. "
            
            # Model-specific strict prompts for quick mode
            if mode == "quick":
                if model == "claude":
                    analysis_prompt = (
                        f"{lang_context}Analyze the following code and return up to 2 bugs and 2 suggestions, in valid JSON. "
                        "Do NOT return code, markdown, or code blocks. Only return a short JSON analysis. "
                        "If you see code, do not repeat it, only analyze.\nCode:\n{code}"
                    ).replace("{code}", code)
                    doc_prompt = (
                        f"{lang_context}Summarize what this code does in 2-3 sentences. "
                        "Do NOT return code, markdown, or code blocks. Only return a summary."
                    )
                elif model == "mercury":
                    analysis_prompt = (
                        f"{lang_context}Analyze the following code and return up to 2 bugs and 2 suggestions, in valid JSON. "
                        "Do NOT return code, markdown, or code blocks. Only return a short JSON analysis. "
                        "If you see code, do not repeat it, only analyze.\nCode:\n{code}"
                    ).replace("{code}", code)
                    doc_prompt = (
                        f"{lang_context}Summarize what this code does in 2-3 sentences. "
                        "Do NOT return code, markdown, or code blocks. Only return a summary."
                    )
                else:  # deepseek or others
                    analysis_prompt = f"{lang_context}Analyze the following code and return up to 2 bugs and 2 suggestions, in valid JSON. Be concise.\nCode:\n{code}"
                    doc_prompt = f"{lang_context}Summarize what this code does in 2-3 sentences."
                max_tokens = 600
                do_fixes = False
            else:
                analysis_prompt = f"{lang_context}" + self.prompts['analysis'].format(code=code)
                doc_prompt = f"{lang_context}" + self.prompts['documentation'].format(code=code)
                max_tokens = 2000
                do_fixes = True
            
            # Retry logic for LLM calls
            def call_llm_with_retry(prompt, max_attempts=3):
                for attempt in range(max_attempts):
                    try:
                        response = llm.invoke(prompt)
                        return response
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise e
                        logging.warning(f'LLM call attempt {attempt + 1} failed: {e}, retrying...')
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            # Parallelize LLM calls with retry
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_analysis = executor.submit(lambda: call_llm_with_retry(analysis_prompt))
                future_doc = executor.submit(lambda: call_llm_with_retry(doc_prompt))
                
                try:
                    result = future_analysis.result()
                    doc_result = future_doc.result()
                except Exception as e:
                    logging.error(f'LLM calls failed after retries: {e}')
                    raise
            
            analysis_result = result.content if hasattr(result, 'content') else str(result)
            doc_result = doc_result.content if hasattr(doc_result, 'content') else str(doc_result)
            
            # Post-processing for Claude/Mercury quick mode
            fallback_message = "This model could not provide a concise analysis. Try DeepSeek or use Thorough mode."
            def is_code_or_markdown(text):
                if not text:
                    return False
                # Heuristic: code block, markdown, or just code
                if text.strip().startswith("```") or re.match(r"^[ \t]*function|def |let |const |class |public |private |#include|import |package |var |<|</", text.strip()):
                    return True
                if '```' in text or text.strip().startswith('#'):
                    return True
                return False
            try:
                parsed_result = parse_llm_response(analysis_result)
                # If Claude/Mercury returns code/markdown or irrelevant, fallback
                if mode == "quick" and model in ("claude", "mercury"):
                    # If the result is just code, markdown, or an irrelevant message, fallback
                    if (is_code_or_markdown(analysis_result) or
                        (isinstance(parsed_result, dict) and not parsed_result.get("potential_bugs") and not parsed_result.get("improvement_suggestions"))):
                        parsed_result = {
                            "code_quality_score": 0,
                            "potential_bugs": [fallback_message],
                            "improvement_suggestions": [],
                            "documentation": fallback_message
                        }
            except Exception as e:
                logging.warning(f'Failed to parse LLM response: {e}')
                parsed_result = {
                    "code_quality_score": 50,
                    "potential_bugs": [fallback_message],
                    "improvement_suggestions": [],
                    "documentation": fallback_message
                }
            # Post-process doc_result for Claude/Mercury
            if mode == "quick" and model in ("claude", "mercury") and is_code_or_markdown(doc_result):
                doc_result = fallback_message
            
            # In thorough mode, always fill empty fields with defaults
            if mode == "thorough":
                if not parsed_result.get("potential_bugs") or len(parsed_result["potential_bugs"]) == 0:
                    parsed_result["potential_bugs"] = ["No bugs detected."]
                if not parsed_result.get("improvement_suggestions") or len(parsed_result["improvement_suggestions"]) == 0:
                    parsed_result["improvement_suggestions"] = ["No suggestions."]
                if not doc_result or doc_result.strip() == "":
                    doc_result = "No documentation generated."
            
            # Fix suggestions only in thorough mode
            if do_fixes and hasattr(self, 'fix_generator') and self.fix_generator:
                try:
                    issues = []
                    for i, bug in enumerate(parsed_result.get("potential_bugs", [])):
                        issues.append({
                            'type': 'bug',
                            'description': bug,
                            'line_number': 0,
                            'code_snippet': code[:200] + '...' if len(code) > 200 else code,
                            'severity': 'medium'
                        })
                    for i, suggestion in enumerate(parsed_result.get("improvement_suggestions", [])):
                        if not suggestion.startswith('RAG:'):
                            issues.append({
                                'type': 'improvement',
                                'description': suggestion,
                                'line_number': 0,
                                'code_snippet': code[:200] + '...' if len(code) > 200 else code,
                                'severity': 'low'
                            })
                    fix_suggestions = self.fix_generator.generate_fix_suggestions(
                        code, issues, detected_language
                    )
                    result_obj.fix_suggestions = fix_suggestions
                except Exception as e:
                    logging.warning(f'Fix suggestions failed: {e}')
                    result_obj.fix_suggestions = []
            else:
                result_obj.fix_suggestions = []
            
            return result_obj
            
        except Exception as e:
            logging.error(f'Analysis failed: {e}')
            # Always return a valid result object even on failure
            return CodeAnalysisResult(
                code_quality_score=0,
                potential_bugs=[f"Analysis failed: {str(e)}"],
                improvement_suggestions=[],
                documentation=f"Error during analysis: {str(e)}",
                model_name=model,
                execution_time=0,
                fix_suggestions=[]
            )

    def analyze_with_all_models(self, code: str, mode: str = "quick") -> Dict[str, CodeAnalysisResult]:
        """Analyze code using all available models."""
        results = {}
        for model_name in self.models.keys():
            try:
                results[model_name] = self.analyze_code(code, model=model_name, mode=mode)
            except Exception as e:
                print(f"Error analyzing with {model_name}: {e}")
                results[model_name] = CodeAnalysisResult(
                    code_quality_score=0,
                    potential_bugs=[f"Error: {str(e)}"],
                    improvement_suggestions=[],
                    documentation="",
                    model_name=model_name,
                    execution_time=0
                )
        return results

    def get_model_comparison(self) -> Dict[str, ModelEvaluationResult]:
        """Get comparison of model performances."""
        return self.evaluator.get_model_comparison()

    def generate_report(self, analysis_result: CodeAnalysisResult) -> str:
        """Generate a formatted report from analysis results."""
        report = f"""
Code Analysis Report
===================

Quality Score: {analysis_result.code_quality_score}/100
Execution Time: {analysis_result.execution_time:.2f} seconds

Potential Bugs:
{chr(10).join(f"- {bug}" for bug in analysis_result.potential_bugs) if analysis_result.potential_bugs else "- None detected"}

Improvement Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in analysis_result.improvement_suggestions) if analysis_result.improvement_suggestions else "- None"}

Documentation:
{analysis_result.documentation}
"""
        return report