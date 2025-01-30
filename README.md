LLM Code Analyzer

An intelligent code analysis system that leverages multiple Large Language Models (LLMs) to provide comprehensive code reviews, documentation generation, and quality assessments.
🚀 Features

Multi-Model Analysis: Integrates with OpenAI GPT and Anthropic Claude
Automated Code Review: Analyzes code quality, identifies potential bugs
Smart Documentation: Generates comprehensive documentation using chain-of-thought prompting
Performance Comparison: Evaluates and compares different LLM models' performance
Custom Prompting: Implements strategic prompting for improved accuracy

🛠️ Installation

Clone the repository:

bashCopygit clone https://github.com/arun3676/llm-code-analyzer.git
cd llm-code-analyzer

Create a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt

Set up your environment variables:

bashCopy# Create .env file and add your API keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
📚 Usage
pythonCopyfrom code_analyzer import CodeAnalyzer

# Initialize analyzer
analyzer = CodeAnalyzer()

# Analyze code
code = """
def greet(name):
    print(f"Hello, {name}!")
"""

# Get analysis results
result = analyzer.analyze_code(code, model="gpt")
print(analyzer.generate_report(result))
📊 Project Structure
Copyllm-code-analyzer/
├── code_analyzer/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── prompts.py
│   ├── config.py
│   ├── evaluator.py
│   └── utils.py
├── notebooks/
│   └── examples/
├── tests/
├── requirements.txt
└── README.md
🧪 Running Tests
bashCopypytest tests/
📝 Documentation
Check out the Jupyter notebooks in the notebooks/ directory for detailed examples and usage scenarios.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
✨ Acknowledgments

OpenAI GPT models
Anthropic Claude
LangChain framework

👤 Author
Arun Kumar Chukkala

Email: arunkiran721@gmail.com
GitHub: @arun3676
