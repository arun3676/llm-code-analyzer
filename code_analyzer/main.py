from typing import Dict, List, Optional, Any
import time
import os
from functools import wraps
from dotenv import load_dotenv
import traceback
import concurrent.futures
import re

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

# Import RAG features with error handling
try:
    from .rag_assistant import RAGCodeAssistant
    RAG_AVAILABLE = True
except ImportError:
    print("Warning: RAG features not available")
    RAG_AVAILABLE = False

# Import fix suggestion generator
try:
    from .fix_suggestions import FixSuggestionGenerator
    FIX_SUGGESTIONS_AVAILABLE = True
except ImportError:
    print("Warning: Fix suggestions not available")
    FIX_SUGGESTIONS_AVAILABLE = False

# Import language detector
try:
    from .language_detector import LanguageDetector
    LANGUAGE_DETECTOR_AVAILABLE = True
except ImportError:
    print("Warning: Language detector not available")
    LANGUAGE_DETECTOR_AVAILABLE = False

# Import new analyzers
try:
    from .framework_analyzer import FrameworkAnalyzer
    FRAMEWORK_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: Framework analyzer not available")
    FRAMEWORK_ANALYZER_AVAILABLE = False

try:
    from .cloud_analyzer import CloudAnalyzer
    CLOUD_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: Cloud analyzer not available")
    CLOUD_ANALYZER_AVAILABLE = False

try:
    from .container_analyzer import ContainerAnalyzer
    CONTAINER_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: Container analyzer not available")
    CONTAINER_ANALYZER_AVAILABLE = False

# DeepSeek wrapper using OpenAI SDK
class DeepSeekWrapper:
    def __init__(self, api_key, model_name="deepseek-chat", temperature=0.1):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, prompt):
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

# Mercury wrapper using OpenAI SDK
class MercuryWrapper:
    def __init__(self, api_key, model_name="mercury-coder", temperature=0.1):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.inceptionlabs.ai/v1")
        self.model_name = model_name
        self.temperature = temperature
    def invoke(self, prompt):
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

# Set default model names
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_CLAUDE_MODEL = "claude-3-haiku-20240307"
DEFAULT_MERCURY_MODEL = "mercury"

class CodeAnalyzer:
    """Main class for analyzing code using various LLM models with integrated RAG capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_rag: bool = True):
        """Initialize the code analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG
        self.evaluator = ModelEvaluator()
        self.enable_rag = enable_rag and RAG_AVAILABLE
        
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
        
        # Check for API keys
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        mercury_api_key = os.getenv("MERCURY_API_KEY")
        
        if not deepseek_api_key:
            print("WARNING: DEEPSEEK_API_KEY not found in environment variables")
        else:
            print(f"Found DeepSeek API key: {deepseek_api_key[:5]}...{deepseek_api_key[-4:]}")
        
        if not anthropic_api_key:
            print("WARNING: ANTHROPIC_API_KEY not found in environment variables")
        else:
            print(f"Found Anthropic API key: {anthropic_api_key[:5]}...{anthropic_api_key[-4:]}")
        
        if not mercury_api_key:
            print("WARNING: MERCURY_API_KEY not found in environment variables")
        else:
            print(f"Found Mercury API key: {mercury_api_key[:5]}...{mercury_api_key[-4:]}")
        
        # Initialize LLM clients
        self.models = {}
        
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
        
        # Initialize RAG assistant
        self.rag_assistant = None
        if self.enable_rag:
            try:
                print("Initializing RAG Code Assistant...")
                # Use current directory as codebase path
                codebase_path = os.getcwd()
                self.rag_assistant = RAGCodeAssistant(codebase_path=codebase_path)
                
                # Index the codebase
                print("Indexing codebase...")
                self.rag_assistant.index_codebase()
                print("RAG Code Assistant initialized successfully!")
            except Exception as e:
                print(f"Error initializing RAG assistant: {e}")
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
    def analyze_code(self, code: str, model: str = "deepseek", include_rag_suggestions: bool = True, file_path: Optional[str] = None, language: Optional[str] = None, mode: str = "quick") -> CodeAnalysisResult:
        """
        Analyze code using specified LLM model with optional RAG suggestions and quick/thorough mode.
        
        Args:
            code: Source code to analyze
            model: Model to use ('deepseek' or 'claude')
            include_rag_suggestions: Whether to include RAG-based suggestions
            file_path: Path to the file containing the code
            language: Optional language of the code
            mode: Analysis mode ('quick' or 'thorough')
                - 'quick': Fast analysis with 2 bugs/suggestions, 600 tokens
                - 'thorough': Optimized for 4GB memory - 5 bugs/suggestions, 1200 tokens, RAG enabled with memory optimizations
            
        Returns:
            CodeAnalysisResult object containing analysis results
        """
        if model not in self.models:
            raise ValueError(f"Unsupported model: {model}. Available models: {list(self.models.keys())}")
            
        # Always use override if set, else detect
        detected_language = language
        if not detected_language and self.language_detector:
            lang_info = self.language_detector.detect_language(code, file_path)
            detected_language = lang_info.name
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
                    f"{lang_context}Analyze the following code and return up to 1 bug and 1 suggestion, in valid JSON. "
                    "Do NOT return code, markdown, or code blocks. Only return a short JSON analysis. "
                    "If you see code, do not repeat it, only analyze.\nCode:\n{code}"
                ).replace("{code}", code)
                doc_prompt = (
                    f"{lang_context}Summarize what this code does in 1-2 sentences. "
                    "Do NOT return code, markdown, or code blocks. Only return a summary."
                )
            elif model == "mercury":
                analysis_prompt = (
                    f"{lang_context}Analyze the following code and return up to 1 bug and 1 suggestion, in valid JSON. "
                    "Do NOT return code, markdown, or code blocks. Only return a short JSON analysis. "
                    "If you see code, do not repeat it, only analyze.\nCode:\n{code}"
                ).replace("{code}", code)
                doc_prompt = (
                    f"{lang_context}Summarize what this code does in 1-2 sentences. "
                    "Do NOT return code, markdown, or code blocks. Only return a summary."
                )
            else:  # deepseek or others
                analysis_prompt = f"{lang_context}Analyze the following code and return up to 1 bug and 1 suggestion, in valid JSON. Be concise.\nCode:\n{code}"
                doc_prompt = f"{lang_context}Summarize what this code does in 1-2 sentences."
            max_tokens = 400
            analysis_timeout = 20
            do_rag = False
            do_fixes = False
            do_specialized = False
        elif mode == "thorough":
            # Optimized thorough mode for 4GB memory - RAG enabled but with memory optimizations
            analysis_prompt = f"{lang_context}Analyze the following code thoroughly and return up to 5 bugs and 5 suggestions, in valid JSON. Be comprehensive but concise.\nCode:\n{code}"
            doc_prompt = f"{lang_context}Provide detailed documentation for this code, explaining its purpose, parameters, and usage in 3-4 sentences."
            max_tokens = 1200  # Reduced from 2000
            do_rag = True  # Re-enabled RAG but with optimizations
            do_fixes = False  # Disabled for memory optimization
            do_specialized = False  # Disabled for memory optimization
        else:
            analysis_prompt = f"{lang_context}" + self.prompts['analysis'].format(code=code)
            doc_prompt = f"{lang_context}" + self.prompts['documentation'].format(code=code)
            max_tokens = 2000
            do_rag = include_rag_suggestions
            do_fixes = True
            do_specialized = True
        
        # Parallelize LLM calls for quick mode, sequential for thorough mode to save memory
        if mode == "thorough":
            # Sequential calls for memory optimization
            result = llm.invoke(analysis_prompt)
            doc_result = llm.invoke(doc_prompt)
        else:
            # Parallel calls for other modes
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_analysis = executor.submit(lambda: llm.invoke(analysis_prompt))
                future_doc = executor.submit(lambda: llm.invoke(doc_prompt))
                result = future_analysis.result()
                doc_result = future_doc.result()
        
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
        
        # RAG and fix suggestions only in thorough mode
        rag_suggestions = []
        if do_rag and self.rag_assistant:
            try:
                # Memory-optimized RAG: limit suggestions and use smaller search scope
                rag_suggestions = self.rag_assistant.get_code_suggestions(
                    code, detected_language, "Looking for similar patterns and improvements"
                )
                rag_count = 0
                max_rag_suggestions = 2 if mode == "thorough" else 3  # Limit for memory optimization
                for suggestion in rag_suggestions:
                    if rag_count >= max_rag_suggestions:
                        break
                    explanation = suggestion.get('explanation', '')
                    if explanation:
                        short_expl = explanation.split('This code snippet')[0].strip()
                        if len(short_expl) > 150:  # Reduced from 180 for memory optimization
                            short_expl = short_expl[:150] + '...'
                        parsed_result["improvement_suggestions"].append(f"RAG: {short_expl}")
                        rag_count += 1
            except Exception as e:
                print(f"RAG error (continuing without RAG): {e}")
                pass
        
        # Run specialized analyzers
        framework_analysis = None
        cloud_analysis = None
        container_analysis = None
        
        if do_specialized and self.framework_analyzer and file_path:
            try:
                framework_analysis = self.framework_analyzer.analyze_code(file_path, code)
            except Exception as e:
                print(f"Framework analysis error: {e}")
        
        if do_specialized and self.cloud_analyzer and file_path:
            try:
                cloud_analysis = self.cloud_analyzer.analyze_code(file_path, code)
            except Exception as e:
                print(f"Cloud analysis error: {e}")
        
        if do_specialized and self.container_analyzer and file_path:
            try:
                container_analysis = self.container_analyzer.analyze_code(file_path, code)
            except Exception as e:
                print(f"Container analysis error: {e}")
        
        # Add specialized analysis results to suggestions
        if framework_analysis and framework_analysis.get('issues'):
            for issue in framework_analysis['issues']:
                parsed_result["improvement_suggestions"].append(
                    f"Framework ({framework_analysis['framework']}): {issue['message']}"
                )
        
        if cloud_analysis and cloud_analysis.get('issues'):
            for issue in cloud_analysis['issues']:
                parsed_result["improvement_suggestions"].append(
                    f"Cloud ({cloud_analysis['platform']}): {issue['message']}"
                )
        
        if container_analysis and container_analysis.get('issues'):
            for issue in container_analysis['issues']:
                parsed_result["improvement_suggestions"].append(
                    f"Container ({container_analysis['config_type']}): {issue['message']}"
                )
        
        # Create result object
        result_obj = CodeAnalysisResult(
            code_quality_score=parsed_result.get("code_quality_score", 50),
            potential_bugs=parsed_result.get("potential_bugs", []),
            improvement_suggestions=parsed_result.get("improvement_suggestions", []),
            documentation=doc_result,
            model_name=model,
            execution_time=time.time()  # This will be set by the timer decorator
        )
        
        # Fix suggestions only in thorough mode
        if do_fixes and self.fix_generator:
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
                result_obj.fix_suggestions = []
        else:
            result_obj.fix_suggestions = []
        
        # Get config values for timeout and memory
        analysis_timeout = self.config.get('analysis', {}).get('timeout', 120)
        max_memory_mb = self.config.get('analysis', {}).get('max_memory_mb', 4096)
        
        # Optional: Warn if memory usage exceeds max_memory_mb
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if mem_mb > max_memory_mb * 0.95:
                print(f"Warning: Memory usage is high ({mem_mb:.1f} MB / {max_memory_mb} MB). Analysis may be unstable.")
        except ImportError:
            pass
        
        return result_obj

    def search_similar_code(self, query: str, top_k: int = 5, language_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar code in the codebase using RAG.
        
        Args:
            query: Search query
            top_k: Number of results to return
            language_filter: Optional language filter
            
        Returns:
            List of search results
        """
        if not self.rag_assistant:
            return []
        
        try:
            results = self.rag_assistant.search_code(query, top_k, language_filter)
            return [{
                'snippet': {
                    'content': result.snippet.content,
                    'file_path': result.snippet.file_path,
                    'language': result.snippet.language,
                    'function_name': result.snippet.function_name,
                    'class_name': result.snippet.class_name
                },
                'relevance_score': result.relevance_score,
                'context': result.context,
                'explanation': result.explanation
            } for result in results]
        except Exception as e:
            print(f"Error searching code: {e}")
            return []
    
    def get_codebase_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase."""
        if not self.rag_assistant:
            return {'error': 'RAG assistant not available'}
        
        try:
            return self.rag_assistant.get_codebase_stats()
        except Exception as e:
            return {'error': f'Failed to get stats: {str(e)}'}
    
    def reindex_codebase(self, force_reindex: bool = True) -> int:
        """
        Reindex the codebase for RAG search.
        
        Args:
            force_reindex: Whether to force reindexing
            
        Returns:
            Number of snippets indexed
        """
        if not self.rag_assistant:
            return 0
        
        try:
            return self.rag_assistant.index_codebase(force_reindex=force_reindex)
        except Exception as e:
            print(f"Error reindexing codebase: {e}")
            return 0

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