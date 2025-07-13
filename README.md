# LLM Code Analyzer

Advanced AI-powered code analysis tool with multiple model support, RAG capabilities, and comprehensive code quality assessment.

## Features

- **Multi-Model Support**: OpenAI GPT-4, Anthropic Claude, DeepSeek, and Mercury
- **RAG Integration**: Context-aware code suggestions using your codebase
- **Advanced Analysis**: Security, performance, and framework-specific analysis
- **Web Interface**: Modern Flask-based web UI
- **Mock Testing**: Built-in mock mode for testing without API keys
- **Repository Q&A**: Ask questions about your codebase using RAG

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd llm-code-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Copy the environment example file and fill in your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your API keys:
```env
# Required for OpenAI models (GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-...

# Required for Anthropic Claude models
ANTHROPIC_API_KEY=sk-ant-...

# Required for DeepSeek models
DEEPSEEK_API_KEY=sk-...

# Required for Mercury models
MERCURY_API_KEY=sk_...

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=1
```

### 4. Run the Application

#### Web Interface (Local Development)
```bash
# Option 1: Using Flask directly
flask run --debug

# Option 2: Using the run script
python run_local.py

# Option 3: Using debug script
python debug_run.py
```

Then open http://localhost:5000 in your browser.

#### Command Line
```bash
python debug_run.py
```

#### Mock Mode (for testing without API keys)
```bash
python -c "from code_analyzer.main import CodeAnalyzer; analyzer = CodeAnalyzer(mock=True); result = analyzer.analyze_code('def test(): pass', model='mock'); print(result)"
```

## API Endpoints

### Analyze Code
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(): print(\"world\")",
    "model": "deepseek",
    "mode": "quick"
  }'
```

### Ask About Repository
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the code analyzer work?"
  }'
```

## Testing

### Run Performance Tests (Mock Mode)
```bash
python tests/test_performance.py
```

### Test Imports
```bash
python test_imports.py
```

### Check Dependencies
```bash
python install_deps.py
```

## Deployment

### Local Development
```bash
flask run --debug
```

### Docker
```bash
docker build -t llm-code-analyzer .
docker run -p 8080:8080 llm-code-analyzer
```

### Google Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud run deploy llm-code-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=your_key,ANTHROPIC_API_KEY=your_key,DEEPSEEK_API_KEY=your_key,MERCURY_API_KEY=your_key"
```

### Heroku
```bash
# Deploy to Heroku
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
heroku config:set ANTHROPIC_API_KEY=your_key
heroku config:set DEEPSEEK_API_KEY=your_key
heroku config:set MERCURY_API_KEY=your_key
git push heroku main
```

## API Key Requirements

The application requires API keys for the following services:

- **OpenAI**: For GPT-4 and GPT-3.5 models
- **Anthropic**: For Claude models
- **DeepSeek**: For DeepSeek models
- **Mercury**: For Mercury models

If you don't have API keys, you can:
1. Use mock mode for testing: `CodeAnalyzer(mock=True)`
2. Get API keys from the respective platforms:
   - [OpenAI Platform](https://platform.openai.com/api-keys)
   - [Anthropic Console](https://console.anthropic.com/)
   - [DeepSeek Platform](https://platform.deepseek.com/)
   - [Mercury Platform](https://mercury.ai/)

## Error Handling

The application includes comprehensive error handling:

- **Missing API Keys**: Clear error messages with instructions
- **Network Issues**: Graceful fallbacks and retry logic
- **Import Errors**: Fallback to basic mode with warnings
- **Analysis Failures**: Detailed error reporting
- **Input Validation**: Proper validation of all inputs
- **CORS Support**: Cross-origin request handling

## Development

### Project Structure
```
llm-code-analyzer/
├── code_analyzer/          # Main application package
│   ├── main.py            # Core analyzer
│   ├── advanced_analyzer.py # Advanced features
│   ├── web/               # Flask web interface
│   └── ...
├── tests/                 # Test files
├── requirements.txt       # Dependencies
├── env_example.txt        # Environment template
├── Dockerfile            # Docker configuration
└── README.md             # This file
```

### Adding New Models
1. Add API key validation in `main.py`
2. Create model wrapper class
3. Register in `CodeAnalyzer.__init__()`
4. Update environment template

### Mock Mode
Use mock mode for testing without API keys:
```python
analyzer = CodeAnalyzer(mock=True)
result = analyzer.analyze_code(code, model='mock')
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Copy `env_example.txt` to `.env` and fill in your keys
2. **Import Errors**: Run `python install_deps.py` to check dependencies
3. **Network Issues**: Check your internet connection and API key validity
4. **Port Conflicts**: Change the port in `run_local.py` or `debug_run.py`
5. **CORS Issues**: Ensure `flask-cors` is installed for cross-origin requests

### Debug Mode
Run with debug logging:
```bash
python debug_run.py
```

### Cloud Run Issues
- Ensure `PORT` environment variable is set to 8080
- Check that all API keys are configured in environment variables
- Verify the Dockerfile is properly configured

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
