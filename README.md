# LLM Code Analyzer

AI-powered code analysis system using multiple LLMs with advanced features including performance profiling, security analysis, and RAG-powered code search.

## 🚀 Features

### Core Analysis
- **Multi-LLM Support**: Analyze code with DeepSeek, Claude, OpenAI, and Mercury
- **Code Quality Assessment**: Get quality scores, bug detection, and improvement suggestions
- **Documentation Generation**: Auto-generate comprehensive code documentation

### 🔥 NEW: Performance Profiling & Optimization Suggestions
- **Performance Analysis**: Detect performance bottlenecks and anti-patterns
- **AI-Powered Optimizations**: Get LLM-generated optimization suggestions with code examples
- **Function Profiling**: Profile specific functions with execution time and memory usage
- **Benchmarking**: Compare different code implementations for performance
- **Algorithm Complexity Analysis**: Analyze time and space complexity
- **Memory Profiling**: Track memory usage during function execution

### Advanced Features
- **Security Analysis**: Detect security vulnerabilities and best practices
- **RAG-Powered Search**: Search your codebase with semantic understanding
- **GitHub Integration**: Analyze code directly from GitHub repositories
- **Multi-Modal Analysis**: Analyze code in images, screenshots, and diagrams
- **Comprehensive Reporting**: Generate detailed reports in JSON, HTML, and text formats

### 🆕 NEW: Framework-Specific Analysis
- **React/Angular Analysis**: Detect React hooks misuse, missing key props, and Angular patterns
- **Django Analysis**: Identify raw SQL queries, CSRF issues, and Django best practices
- **Spring Analysis**: Find field injection issues, missing transactions, and Spring patterns

### 🆕 NEW: Cloud Platform Integration Analysis
- **AWS Analysis**: Detect hardcoded credentials, missing error handling, and S3 security issues
- **Azure Analysis**: Find connection string issues, missing retry policies, and Azure best practices
- **GCP Analysis**: Identify project ID issues, authentication problems, and GCP patterns

### 🆕 NEW: Container & Kubernetes Analysis
- **Dockerfile Analysis**: Detect root user, missing health checks, and Docker best practices
- **Kubernetes Analysis**: Find missing resource limits, security contexts, and K8s patterns
- **Security Scanning**: Identify container security vulnerabilities and misconfigurations

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-code-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
MERCURY_API_KEY=your_mercury_api_key
```

## 🎯 Usage

### Web Interface
Start the web server:
```bash
python -m code_analyzer.web.app
```

Visit `http://localhost:5000` to access the web interface with all features including the new Performance Analysis tab.

### Command Line
```python
from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig

# Configure with performance analysis enabled
config = AnalysisConfig(
    enable_performance=True,
    performance_analysis_level='comprehensive'
)

analyzer = AdvancedCodeAnalyzer(config)

# Analyze code with performance profiling
result = analyzer.analyze_code_advanced(
    code="your code here",
    language="python",
    file_path="example.py"
)

# Access performance analysis results
if result.performance_report:
    print(f"Performance Score: {result.performance_report.overall_score}/100")
    for issue in result.performance_report.issues:
        print(f"Issue: {issue.issue_type} - {issue.description}")
```

### Framework, Cloud & Container Analysis Examples

#### Framework-Specific Analysis
```python
from code_analyzer.main import CodeAnalyzer

analyzer = CodeAnalyzer()

# React code with issues
react_code = """
import React, { useState } from 'react';

function MyComponent() {
    const [data, setData] = useState(null);
    fetch('/api/data').then(res => setData(res.json()));  // Missing useEffect
    
    return (
        <div>
            {data && data.map(item => (
                <div>{item.name}</div>  // Missing key prop
            ))}
        </div>
    );
}
"""

result = analyzer.analyze_code(
    code=react_code,
    file_path="MyComponent.jsx",
    mode="thorough"
)

# Framework-specific suggestions will be included
for suggestion in result.improvement_suggestions:
    if suggestion.startswith('Framework'):
        print(suggestion)
```

#### Cloud Platform Analysis
```python
# AWS code with issues
aws_code = """
import boto3

aws_access_key_id = "AKIA..."  # Hardcoded credentials
s3 = boto3.client('s3')
s3.upload_file('file.txt', 'bucket', 'file.txt')  # No error handling
"""

result = analyzer.analyze_code(
    code=aws_code,
    file_path="aws_upload.py",
    mode="thorough"
)

# Cloud-specific suggestions will be included
for suggestion in result.improvement_suggestions:
    if suggestion.startswith('Cloud'):
        print(suggestion)
```

#### Container Analysis
```python
# Dockerfile with issues
dockerfile_content = """
FROM python:latest
COPY . /app
USER root
CMD ["python", "app.py"]
"""

result = analyzer.analyze_code(
    code=dockerfile_content,
    file_path="Dockerfile",
    mode="thorough"
)

# Container-specific suggestions will be included
for suggestion in result.improvement_suggestions:
    if suggestion.startswith('Container'):
        print(suggestion)
```

### Performance Analysis Examples

#### Basic Performance Analysis
```python
from code_analyzer.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Analyze code for performance issues
report = analyzer.analyze_code_performance(code, 'python', 'file.py')
print(f"Performance Score: {report.overall_score}/100")
```

#### Function Profiling
```python
def my_function(n):
    return sum(i**2 for i in range(n))

# Profile function performance
profile_result = analyzer.profile_function(my_function, 10000)
print(f"Execution Time: {profile_result['execution_time']}s")
print(f"Profile Stats: {profile_result['profile_stats']}")
```

#### Benchmarking Alternatives
```python
def version1(n): return [i**2 for i in range(n)]
def version2(n): return list(i**2 for i in range(n))

code_versions = [("List Comp", version1), ("Generator", version2)]
benchmark_result = analyzer.benchmark_alternatives(code_versions, 10000)
print(f"Fastest: {benchmark_result['fastest']}")
```

## 📊 Performance Analysis Features

### Detected Issues
- **Nested Loops**: O(n²) or worse complexity
- **Inefficient List Operations**: Append in loops, list concatenation
- **String Concatenation**: Inefficient string building in loops
- **Memory Inefficient Operations**: Large range operations
- **Inefficient Data Structures**: List lookups instead of sets/dicts
- **Recursive Functions**: Without memoization

### AI-Powered Optimizations
- **Code Examples**: Before/after optimization comparisons
- **Performance Estimates**: Expected improvement percentages
- **Best Practices**: Language-specific optimization recommendations
- **Complexity Analysis**: Time and space complexity breakdown

### Profiling Capabilities
- **Execution Time**: Precise timing measurements
- **Memory Usage**: Memory consumption tracking
- **cProfile Integration**: Detailed function-level profiling
- **Statistical Analysis**: Performance statistics and comparisons

## 🔧 Configuration

### Performance Analysis Levels
- **basic**: Pattern-based analysis only
- **standard**: Pattern + AST analysis (default)
- **comprehensive**: Full analysis with AI-powered suggestions

### Advanced Configuration
```python
config = AnalysisConfig(
    enable_performance=True,
    performance_analysis_level='comprehensive',
    enable_security=True,
    enable_rag=True,
    enable_multimodal=True
)
```

## 🧪 Testing

Run the performance analysis test:
```bash
python test_performance.py
```

This will demonstrate all the new performance profiling features with sample code.

Run the new features test:
```bash
python test_new_features.py
```

This will demonstrate the new framework, cloud, and container analysis features with examples.

## 📈 Performance Analysis Examples

### Example 1: Detecting Nested Loops
```python
# Inefficient code
for i in range(n):
    for j in range(n):
        result.append(matrix[i][j])

# AI suggests:
result = [matrix[i][j] for i in range(n) for j in range(n)]
# Estimated improvement: 40% faster, 30% less memory
```

### Example 2: String Concatenation
```python
# Inefficient code
result = ""
for i in range(1000):
    result += str(i)

# AI suggests:
result = "".join(str(i) for i in range(1000))
# Estimated improvement: 60% faster, 50% less memory
```

### Example 3: Inefficient Lookups
```python
# Inefficient code
items = [1, 2, 3, 4, 5]
for i in range(1000):
    if i in items:  # O(n) lookup
        pass

# AI suggests:
items_set = {1, 2, 3, 4, 5}
for i in range(1000):
    if i in items_set:  # O(1) lookup
        pass
# Estimated improvement: 90% faster for large datasets
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain for LLM integration
- Uses ChromaDB for RAG capabilities
- Performance analysis powered by cProfile and psutil
- Web interface built with Flask and modern CSS
