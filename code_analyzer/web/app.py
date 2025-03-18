from flask import Flask, render_template, request, jsonify
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Check API keys before importing
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_key or not anthropic_key:
    print("\n" + "="*50)
    print("API KEY CONFIGURATION ERROR")
    print("="*50)
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Add your OpenAI API key to the .env file.")
    
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("   Add your Anthropic API key to the .env file.")
    
    print("\nCreate a .env file in your project root with:")
    print("OPENAI_API_KEY=your_actual_openai_api_key")
    print("ANTHROPIC_API_KEY=your_actual_anthropic_api_key")
    print("="*50 + "\n")

# Import the CodeAnalyzer after checking keys
from code_analyzer.main import CodeAnalyzer

app = Flask(__name__)

try:
    analyzer = CodeAnalyzer()
    available_models = list(analyzer.models.keys())
    if not available_models:
        print("Warning: No models available. Please check your API keys.")
except Exception as e:
    print(f"Error initializing CodeAnalyzer: {e}")
    available_models = []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', available_models=available_models)

@app.route('/analyze', methods=['POST'])
def analyze_code():
    """Analyze code using selected models."""
    if not available_models:
        return jsonify({'error': 'No API keys configured. Please add your API keys to the .env file.'}), 500
    
    data = request.json
    code = data.get('code', '')
    models = data.get('models', ['gpt'])
    
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    results = {}
    for model in models:
        if model not in available_models:
            results[model] = {'error': f'Model {model} not available. Please check your API keys.'}
            continue
            
        try:
            result = analyzer.analyze_code(code, model=model)
            results[model] = {
                'quality_score': result.code_quality_score,
                'potential_bugs': result.potential_bugs,
                'improvement_suggestions': result.improvement_suggestions,
                'documentation': result.documentation,
                'execution_time': result.execution_time
            }
        except Exception as e:
            results[model] = {'error': str(e)}
    
    return jsonify(results)

@app.route('/comparison')
def get_comparison():
    """Get comparison of model performances."""
    if not available_models:
        return jsonify({'error': 'No API keys configured. Please add your API keys to the .env file.'}), 500
    
    try:
        comparison = analyzer.get_model_comparison()
        result = {}
        
        for model_name, eval_result in comparison.items():
            result[model_name] = {
                'average_quality_score': eval_result.average_quality_score,
                'average_execution_time': eval_result.average_execution_time,
                'success_rate': eval_result.success_rate
            }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Get available models."""
    return jsonify(available_models)

if __name__ == '__main__':
    # Create directory for templates if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    # Create directory for static files if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    if not available_models:
        print("\nWarning: Starting server with no available models.")
        print("The web interface will display an error message.")
        print("Please add your API keys to the .env file and restart the server.\n")
    
    app.run(debug=True)