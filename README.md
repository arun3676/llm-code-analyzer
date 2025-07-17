# LLM Code Analyzer

An advanced AI-powered code analysis tool with support for multiple models, multimodal analysis (code and images), and a modern Streamlit web interface.

## Features

- **Multi-Model Support**: OpenAI GPT-4, Anthropic Claude, DeepSeek, Mercury, and Google Gemini.
- **Multimodal Analysis**: Analyze code snippets and images together to get a comprehensive understanding of your code.
- **Web Interface**: A user-friendly Streamlit web application for easy interaction.
- **RAG Integration**: Context-aware code suggestions using your codebase.
- **Advanced Analysis**: Security, performance, and framework-specific analysis.

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
Create a `.env` file in the root of the project and add your API keys:
```env
# Required for OpenAI models (GPT-4, GPT-3.5)
OPENAI_API_KEY=sk-...

# Required for Anthropic Claude models
ANTHROPIC_API_KEY=sk-ant-...

# Required for DeepSeek models
DEEPSEEK_API_KEY=sk-...

# Required for Mercury models
MERCURY_API_KEY=sk_...

# Required for Google Gemini models
GEMINI_API_KEY=AIza...
```

### 4. Run the Application
```bash
streamlit run code_analyzer/web/app.py
```
Then open the provided URL in your browser.

## Deployment

### Render
The application is configured for easy deployment to Render. Simply connect your GitHub repository to Render and create a new Web Service. Render will automatically detect the `render.yaml` file and configure the service for you.

You will need to add your API keys as environment variables in the Render dashboard.

### Docker
You can also build and run the application using Docker:
```bash
docker build -t llm-code-analyzer .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key -e ANTHROPIC_API_KEY=your_key -e DEEPSEEK_API_KEY=your_key -e MERCURY_API_KEY=your_key -e GEMINI_API_KEY=your_key llm-code-analyzer
```

## Project Structure
```
llm-code-analyzer/
├── code_analyzer/          # Main application package
│   ├── web/               # Streamlit web interface
│   │   └── app.py         # The main Streamlit application
│   ├── advanced_analyzer.py # Advanced analysis features
│   ├── multimodal_analyzer.py # Multimodal analysis features
│   └── ...
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (not committed)
├── Dockerfile             # Docker configuration
├── render.yaml            # Render deployment configuration
└── README.md              # This file
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Make sure you have created a `.env` file and added all the required API keys.
2. **Import Errors**: Ensure you have installed all the dependencies from `requirements.txt`.
3. **Port Conflicts**: If the default port (8501) is in use, you can run the application on a different port:
   ```bash
   streamlit run code_analyzer/web/app.py --server.port <your_port>
   ```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
