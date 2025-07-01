from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
from dotenv import load_dotenv
import json
from pathlib import Path
import tempfile
import traceback
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check API keys before importing
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
mercury_key = os.getenv("MERCURY_API_KEY")

if not openai_key or not anthropic_key or not deepseek_key or not mercury_key:
    print("\n" + "="*50)
    print("API KEY CONFIGURATION ERROR")
    print("="*50)
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Add your OpenAI API key to the .env file.")
    
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("   Add your Anthropic API key to the .env file.")
    
    if not deepseek_key:
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        print("   Add your DeepSeek API key to the .env file.")
    
    if not mercury_key:
        print("❌ MERCURY_API_KEY not found in environment variables")
        print("   Add your Mercury API key to the .env file.")
    
    print("\nCreate a .env file in your project root with:")
    print("OPENAI_API_KEY=your_actual_openai_api_key")
    print("ANTHROPIC_API_KEY=your_actual_anthropic_api_key")
    print("DEEPSEEK_API_KEY=your_actual_deepseek_api_key")
    print("MERCURY_API_KEY=your_actual_mercury_api_key")
    print("="*50 + "\n")

# Import the analyzers after checking keys
from code_analyzer.main import CodeAnalyzer
from code_analyzer.config import DEFAULT_CONFIG

# Import dashboard
try:
    from code_analyzer.dashboard import CodeQualityDashboard
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Dashboard not available: {e}")
    DASHBOARD_AVAILABLE = False

# Import advanced features
try:
    from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

app = Flask(__name__)

# Initialize the code analyzer with RAG enabled
try:
    analyzer = CodeAnalyzer(config=DEFAULT_CONFIG, enable_rag=True)
    print("Advanced analyzer initialized successfully!")
except Exception as e:
    print(f"Error initializing analyzer: {e}")
    traceback.print_exc()
    analyzer = None

# Initialize dashboard if available
dashboard = None
if DASHBOARD_AVAILABLE:
    try:
        dashboard = CodeQualityDashboard()
        print("Code quality dashboard initialized!")
    except Exception as e:
        print(f"Error initializing dashboard: {e}")
        traceback.print_exc()

# Initialize advanced analyzer if available
advanced_analyzer = None
if ADVANCED_FEATURES_AVAILABLE:
    try:
        config = AnalysisConfig(
            enable_rag=True,
            enable_security=True,
            enable_performance=True,
            enable_multimodal=True,
            performance_analysis_level='comprehensive'
        )
        advanced_analyzer = AdvancedCodeAnalyzer(config)
        print("Advanced analyzer with performance profiling initialized!")
    except Exception as e:
        print(f"Error initializing advanced analyzer: {e}")
        traceback.print_exc()

@app.route('/')
def index():
    """Main page with code analysis interface."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_code():
    """Analyze code using the selected model."""
    try:
        data = request.get_json()
        code = data.get('code', '')
        model = data.get('model', 'deepseek')
        language_override = data.get('language')
        file_path = data.get('file_path')
        mode = data.get('mode', 'quick')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        # Analyze code with RAG suggestions enabled and language override
        result = analyzer.analyze_code(code, model=model, include_rag_suggestions=True, file_path=file_path, language=language_override, mode=mode)
        
        # Record analysis in dashboard if available
        if dashboard:
            try:
                detected_language = language_override or 'unknown'
                if hasattr(analyzer, 'language_detector') and analyzer.language_detector:
                    lang_info = analyzer.language_detector.detect_language(code, file_path)
                    detected_language = lang_info.name
                
                dashboard.record_analysis(result, file_path or 'unknown', detected_language, model)
            except Exception as e:
                print(f"Error recording analysis in dashboard: {e}")
        
        # Detect language/frameworks for UI
        detected_language = None
        detected_frameworks = []
        if hasattr(analyzer, 'language_detector') and analyzer.language_detector:
            lang_info = analyzer.language_detector.detect_language(code, file_path)
            detected_language = lang_info.name
            frameworks = analyzer.language_detector.detect_frameworks(code, detected_language, file_path)
            detected_frameworks = [fw.name for fw in frameworks]
        
        # Convert fix suggestions to serializable format
        fix_suggestions = []
        if hasattr(result, 'fix_suggestions') and result.fix_suggestions:
            for fix in result.fix_suggestions:
                fix_suggestions.append({
                    'issue_id': fix.issue_id,
                    'issue_type': fix.issue_type,
                    'severity': fix.severity,
                    'title': fix.title,
                    'description': fix.description,
                    'line_number': fix.line_number,
                    'original_code': fix.original_code,
                    'fixed_code': fix.fixed_code,
                    'explanation': fix.explanation,
                    'confidence': fix.confidence,
                    'tags': fix.tags,
                    'related_links': fix.related_links,
                    'diff': fix.diff,
                    'can_auto_apply': fix.can_auto_apply
                })
        
        return jsonify({
            'quality_score': result.code_quality_score,
            'potential_bugs': result.potential_bugs,
            'improvement_suggestions': result.improvement_suggestions,
            'documentation': result.documentation,
            'execution_time': result.execution_time,
            'model': model,
            'fix_suggestions': fix_suggestions,
            'detected_language': detected_language,
            'detected_frameworks': detected_frameworks
        })
        
    except Exception as e:
        print(f"Error in analyze_code: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_all', methods=['POST'])
def analyze_all_models():
    """Analyze code using all available models."""
    try:
        data = request.get_json()
        code = data.get('code', '')
        mode = data.get('mode', 'quick')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        # Analyze with all models
        results = analyzer.analyze_with_all_models(code, mode=mode)
        
        # Format results for frontend
        formatted_results = {}
        for model_name, result in results.items():
            formatted_results[model_name] = {
                'quality_score': result.code_quality_score,
                'potential_bugs': result.potential_bugs,
                'improvement_suggestions': result.improvement_suggestions,
                'documentation': result.documentation,
                'execution_time': result.execution_time
            }
        
        return jsonify(formatted_results)
        
    except Exception as e:
        print(f"Error in analyze_all_models: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_code():
    """Search for similar code in the codebase using RAG."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        language_filter = data.get('language_filter')
        
        if not query.strip():
            return jsonify({'error': 'No search query provided'}), 400
        
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        # Search using integrated RAG
        results = analyzer.search_similar_code(query, top_k, language_filter)
        
        return jsonify({
            'results': results,
            'query': query,
            'total_results': len(results)
        })
        
    except Exception as e:
        print(f"Error in search_code: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_codebase_stats():
    """Get statistics about the indexed codebase."""
    try:
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        stats = analyzer.get_codebase_stats()
        return jsonify(stats)
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reindex', methods=['POST'])
def reindex_codebase():
    """Reindex the codebase for RAG search."""
    try:
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        snippet_count = analyzer.reindex_codebase(force_reindex=True)
        
        return jsonify({
            'message': f'Successfully indexed {snippet_count} code snippets',
            'snippet_count': snippet_count
        })
        
    except Exception as e:
        print(f"Error reindexing: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    try:
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        
        models = list(analyzer.models.keys())
        return jsonify({'models': models})
        
    except Exception as e:
        print(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vision_models', methods=['GET'])
def get_available_vision_models():
    """Get list of available vision models."""
    try:
        # Import vision analyzer
        try:
            from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
            vision_analyzer = MultiModalAnalyzer()
            models = vision_analyzer.get_available_models()
            return jsonify({'models': models})
        except ImportError:
            return jsonify({'error': 'Multi-modal analysis not available'}), 500
        
    except Exception as e:
        print(f"Error getting vision models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        status = {
            'status': 'healthy',
            'analyzer_initialized': analyzer is not None,
            'rag_available': analyzer.rag_assistant is not None if analyzer else False,
            'models_available': list(analyzer.models.keys()) if analyzer else []
        }
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image using vision-capable LLMs."""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get optional prompt
        prompt = request.form.get('prompt', 'Analyze this image and provide insights about any code, diagrams, or UI elements you can see.')
        model = request.form.get('model', 'gpt4v')
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if not ('.' in image_file.filename and 
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Validate file size (10MB limit)
        if len(image_file.read()) > 10 * 1024 * 1024:
            return jsonify({'error': 'File size must be less than 10MB'}), 400
        
        # Reset file pointer
        image_file.seek(0)
        
        # Import vision analyzer
        try:
            from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
            vision_analyzer = MultiModalAnalyzer()
        except ImportError:
            return jsonify({'error': 'Multi-modal analysis not available. Please install required dependencies.'}), 500
        
        # Analyze image
        import time
        start_time = time.time()
        
        result = vision_analyzer.analyze_image(image_file, prompt, model)
        
        execution_time = time.time() - start_time
        
        return jsonify({
            'analysis': result.get('analysis', ''),
            'code_extracted': result.get('code_extracted', ''),
            'suggestions': result.get('suggestions', []),
            'execution_time': round(execution_time, 2),
            'model': model
        })
        
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_image_all', methods=['POST'])
def analyze_image_all_models():
    """Analyze uploaded image using all available vision models."""
    try:
        # Check if image file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get optional prompt
        prompt = request.form.get('prompt', 'Analyze this image and provide insights about any code, diagrams, or UI elements you can see.')
        
        # Validate file type and size
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if not ('.' in image_file.filename and 
                image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        if len(image_file.read()) > 10 * 1024 * 1024:
            return jsonify({'error': 'File size must be less than 10MB'}), 400
        
        # Reset file pointer
        image_file.seek(0)
        
        # Import vision analyzer
        try:
            from code_analyzer.multimodal_analyzer import MultiModalAnalyzer
            vision_analyzer = MultiModalAnalyzer()
        except ImportError:
            return jsonify({'error': 'Multi-modal analysis not available. Please install required dependencies.'}), 500
        
        # Analyze with all available models
        results = vision_analyzer.analyze_with_all_models(image_file, prompt)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in analyze_image_all: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_github', methods=['POST'])
def analyze_github_link():
    """Analyze code from a GitHub link."""
    try:
        data = request.get_json()
        if not data or 'github_url' not in data:
            return jsonify({'error': 'GitHub URL is required'}), 400
        
        github_url = data['github_url'].strip()
        model = data.get('model', 'deepseek')
        custom_prompt = data.get('prompt', 'Analyze this code and provide a comprehensive explanation.')
        
        # Validate GitHub URL
        if not github_url.startswith(('https://github.com/', 'http://github.com/')):
            return jsonify({'error': 'Please provide a valid GitHub URL'}), 400
        
        # Import GitHub analyzer
        try:
            from code_analyzer.github_analyzer import GitHubAnalyzer
            github_analyzer = GitHubAnalyzer()
        except ImportError:
            return jsonify({'error': 'GitHub analysis not available. Please install required dependencies.'}), 500
        
        # Analyze GitHub link
        import time
        start_time = time.time()
        
        result = github_analyzer.analyze_github_link(github_url, custom_prompt, model)
        
        execution_time = time.time() - start_time
        
        return jsonify({
            'analysis': result.get('analysis', ''),
            'code_content': result.get('code_content', ''),
            'file_info': result.get('file_info', {}),
            'suggestions': result.get('suggestions', []),
            'execution_time': round(execution_time, 2),
            'model': model
        })
        
    except Exception as e:
        print(f"Error in analyze_github_link: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/ask_github_repo', methods=['POST'])
def ask_github_repo():
    """Answer a question about a GitHub repo using DeepSeek and the repo's README."""
    try:
        data = request.get_json()
        github_url = data.get('github_url', '').strip()
        question = data.get('question', '').strip()
        if not github_url or not question:
            return jsonify({'error': 'GitHub URL and question are required'}), 400

        # Parse repo info
        import re
        m = re.match(r'https?://github.com/([^/]+)/([^/]+)', github_url)
        if not m:
            return jsonify({'error': 'Invalid GitHub repo URL'}), 400
        owner, repo = m.group(1), m.group(2)

        # Try to fetch README (try common names)
        readme_content = None
        for readme_name in ['README.md', 'README.MD', 'README.txt', 'README']:
            raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/master/{readme_name}'
            import requests
            resp = requests.get(raw_url)
            if resp.status_code == 200 and resp.text.strip():
                readme_content = resp.text.strip()
                break
            # Try main branch if master fails
            raw_url_main = f'https://raw.githubusercontent.com/{owner}/{repo}/main/{readme_name}'
            resp_main = requests.get(raw_url_main)
            if resp_main.status_code == 200 and resp_main.text.strip():
                readme_content = resp_main.text.strip()
                break
        if not readme_content:
            readme_content = '(No README found in this repository.)'

        # Compose prompt for DeepSeek
        prompt = f"""
You are an expert code assistant. Here is the README of a GitHub repository:

---
{readme_content}
---

Answer the following question about this repository:

"{question}"

Be concise, accurate, and helpful.
"""
        # Use DeepSeek model for answer
        if not analyzer:
            return jsonify({'error': 'Analyzer not initialized'}), 500
        answer_result = analyzer.models['deepseek'].invoke(prompt)
        answer = answer_result.content if hasattr(answer_result, 'content') else str(answer_result)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in ask_github_repo: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_advanced', methods=['POST'])
def analyze_code_advanced():
    """Perform comprehensive code analysis including performance profiling."""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language', 'python')
        model = data.get('model', 'deepseek')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        if not advanced_analyzer:
            return jsonify({'error': 'Advanced analyzer not available'}), 500
        
        # Perform comprehensive analysis
        result = advanced_analyzer.analyze_code_advanced(
            code, language, '', model
        )
        
        # Format performance analysis results
        performance_data = None
        if result.performance_report:
            performance_data = {
                'overall_score': result.performance_report.overall_score,
                'issues': [
                    {
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description,
                        'line_number': issue.line_number,
                        'code_snippet': issue.code_snippet,
                        'impact': issue.impact,
                        'suggestion': issue.suggestion,
                        'ai_optimization': issue.ai_optimization
                    }
                    for issue in result.performance_report.issues
                ],
                'summary': result.performance_report.summary,
                'recommendations': result.performance_report.recommendations,
                'complexity_analysis': result.performance_report.complexity_analysis,
                'ai_insights': result.performance_report.ai_insights,
                'optimization_examples': result.performance_report.optimization_examples
            }
        
        return jsonify({
            'code_analysis': {
                'quality_score': result.code_analysis.code_quality_score if result.code_analysis else None,
                'potential_bugs': result.code_analysis.potential_bugs if result.code_analysis else [],
                'improvement_suggestions': result.code_analysis.improvement_suggestions if result.code_analysis else [],
                'documentation': result.code_analysis.documentation if result.code_analysis else ''
            },
            'performance_analysis': performance_data,
            'security_analysis': result.security_report.__dict__ if result.security_report else None,
            'rag_suggestions': result.rag_suggestions,
            'analysis_timestamp': result.analysis_timestamp,
            'analysis_duration': result.analysis_duration,
            'features_used': result.features_used
        })
        
    except Exception as e:
        print(f"Error in analyze_code_advanced: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/profile_function', methods=['POST'])
def profile_function():
    """Profile a specific function's performance."""
    try:
        data = request.get_json()
        code = data.get('code', '')
        function_name = data.get('function_name', '')
        test_inputs = data.get('test_inputs', [])
        
        if not code.strip() or not function_name:
            return jsonify({'error': 'Code and function name are required'}), 400
        
        if not advanced_analyzer or not advanced_analyzer.performance_analyzer:
            return jsonify({'error': 'Performance analyzer not available'}), 500
        
        # Create a temporary namespace to execute the code
        namespace = {}
        exec(code, namespace)
        
        if function_name not in namespace:
            return jsonify({'error': f'Function {function_name} not found in code'}), 400
        
        func = namespace[function_name]
        
        # Profile the function
        profile_results = []
        for i, test_input in enumerate(test_inputs):
            try:
                if isinstance(test_input, dict):
                    result = advanced_analyzer.performance_analyzer.profile_function(
                        func, **test_input
                    )
                else:
                    result = advanced_analyzer.performance_analyzer.profile_function(
                        func, *test_input
                    )
                
                profile_results.append({
                    'test_case': i + 1,
                    'input': test_input,
                    'execution_time': result['execution_time'],
                    'profile_stats': result['profile_stats']
                })
            except Exception as e:
                profile_results.append({
                    'test_case': i + 1,
                    'input': test_input,
                    'error': str(e)
                })
        
        return jsonify({
            'function_name': function_name,
            'profile_results': profile_results
        })
        
    except Exception as e:
        print(f"Error in profile_function: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/benchmark_alternatives', methods=['POST'])
def benchmark_alternatives():
    """Benchmark different code implementations."""
    try:
        data = request.get_json()
        code_versions = data.get('code_versions', [])  # List of {name, code, function_name}
        test_inputs = data.get('test_inputs', [])
        
        if not code_versions or not test_inputs:
            return jsonify({'error': 'Code versions and test inputs are required'}), 400
        
        if not advanced_analyzer or not advanced_analyzer.performance_analyzer:
            return jsonify({'error': 'Performance analyzer not available'}), 500
        
        benchmark_results = {}
        
        for version in code_versions:
            name = version['name']
            code = version['code']
            function_name = version['function_name']
            
            try:
                # Execute code and get function
                namespace = {}
                exec(code, namespace)
                func = namespace[function_name]
                
                # Benchmark with test inputs
                version_results = []
                for test_input in test_inputs:
                    if isinstance(test_input, dict):
                        result = advanced_analyzer.performance_analyzer.profile_function(
                            func, **test_input
                        )
                    else:
                        result = advanced_analyzer.performance_analyzer.profile_function(
                            func, *test_input
                        )
                    
                    version_results.append({
                        'input': test_input,
                        'execution_time': result['execution_time'],
                        'profile_stats': result['profile_stats']
                    })
                
                benchmark_results[name] = version_results
                
            except Exception as e:
                benchmark_results[name] = {'error': str(e)}
        
        # Find the fastest version for each test case
        fastest_versions = []
        for i, test_input in enumerate(test_inputs):
            fastest = None
            fastest_time = float('inf')
            
            for name, results in benchmark_results.items():
                if 'error' not in results and i < len(results):
                    time = results[i]['execution_time']
                    if time < fastest_time:
                        fastest_time = time
                        fastest = name
            
            fastest_versions.append({
                'test_case': i + 1,
                'fastest': fastest,
                'time': fastest_time
            })
        
        return jsonify({
            'benchmark_results': benchmark_results,
            'fastest_versions': fastest_versions
        })
        
    except Exception as e:
        print(f"Error in benchmark_alternatives: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Dashboard routes
@app.route('/dashboard')
def dashboard_page():
    """Dashboard page with quality trends."""
    return render_template('dashboard.html')

@app.route('/api/dashboard/report', methods=['GET'])
def get_dashboard_report():
    """Get dashboard report data."""
    try:
        if not dashboard:
            return jsonify({'error': 'Dashboard not available'}), 500
        
        days = request.args.get('days', 30, type=int)
        report = dashboard.generate_dashboard_report(days=days)
        
        # Convert dataclass to dict for JSON serialization
        report_dict = {
            'overall_quality_trend': {
                'metric_name': report.overall_quality_trend.metric_name,
                'values': report.overall_quality_trend.values,
                'timestamps': report.overall_quality_trend.timestamps,
                'trend_direction': report.overall_quality_trend.trend_direction,
                'trend_strength': report.overall_quality_trend.trend_strength,
                'average_value': report.overall_quality_trend.average_value,
                'min_value': report.overall_quality_trend.min_value,
                'max_value': report.overall_quality_trend.max_value
            },
            'language_breakdown': report.language_breakdown,
            'model_performance': report.model_performance,
            'top_issues': report.top_issues,
            'improvement_areas': report.improvement_areas,
            'recommendations': report.recommendations,
            'generated_at': report.generated_at,
            'time_period': report.time_period,
            'total_analyses': report.total_analyses
        }
        
        return jsonify(report_dict)
        
    except Exception as e:
        print(f"Error in get_dashboard_report: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/charts', methods=['GET'])
def get_dashboard_charts():
    """Get dashboard chart images."""
    try:
        if not dashboard:
            return jsonify({'error': 'Dashboard not available'}), 500
        
        days = request.args.get('days', 30, type=int)
        charts = dashboard.generate_charts(days=days)
        
        return jsonify(charts)
        
    except Exception as e:
        print(f"Error in get_dashboard_charts: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get raw metrics data."""
    try:
        if not dashboard:
            return jsonify({'error': 'Dashboard not available'}), 500
        
        days = request.args.get('days', 30, type=int)
        language_filter = request.args.get('language')
        model_filter = request.args.get('model')
        
        metrics = dashboard.get_metrics(days=days, language_filter=language_filter, model_filter=model_filter)
        
        # Convert metrics to dict
        metrics_dict = []
        for metric in metrics:
            metrics_dict.append({
                'timestamp': metric.timestamp,
                'file_path': metric.file_path,
                'language': metric.language,
                'quality_score': metric.quality_score,
                'model_name': metric.model_name,
                'execution_time': metric.execution_time,
                'bug_count': metric.bug_count,
                'suggestion_count': metric.suggestion_count,
                'complexity_score': metric.complexity_score,
                'performance_score': metric.performance_score,
                'security_score': metric.security_score,
                'maintainability_score': metric.maintainability_score
            })
        
        return jsonify(metrics_dict)
        
    except Exception as e:
        print(f"Error in get_dashboard_metrics: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create directory for templates if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    # Create directory for static files if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    if not analyzer:
        print("\nWarning: Starting server with no available models.")
        print("The web interface will display an error message.")
        print("Please add your API keys to the .env file and restart the server.\n")
    
    if not ADVANCED_FEATURES_AVAILABLE:
        print("\nWarning: Advanced features are not available.")
        print("Some functionality will be limited.")
        print("Please check your dependencies and configuration.\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)