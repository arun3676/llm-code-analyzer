# LLM Code Analyzer VS Code Extension

A powerful VS Code extension that provides real-time AI-powered code analysis and quality insights directly in your editor.

## Features

- **Real-time Code Analysis**: Analyze your code with multiple AI models (DeepSeek, OpenAI, Anthropic, Mercury)
- **Quality Dashboard**: Track code quality trends over time with beautiful visualizations
- **Issue Detection**: Identify potential bugs and code quality issues
- **Smart Suggestions**: Get AI-powered improvement recommendations
- **Multi-language Support**: Works with Python, JavaScript, Java, C++, and many more languages
- **Context Menu Integration**: Right-click to analyze files or selections
- **Auto-analysis**: Optionally analyze files automatically on save

## Installation

1. Clone this repository
2. Navigate to the `vscode-extension` directory
3. Install dependencies:
   ```bash
   npm install
   ```
4. Compile the extension:
   ```bash
   npm run compile
   ```
5. Press F5 in VS Code to run the extension in a new Extension Development Host window

## Usage

### Commands

- **Analyze Current File**: Analyze the entire active file
- **Analyze Selection**: Analyze only the selected code
- **Quick Analysis**: Perform a quick analysis with default settings
- **Show Quality Dashboard**: Open the web dashboard in your browser

### Context Menu

Right-click in the editor to access:
- Analyze Selection (when text is selected)
- Quick Analysis

### Configuration

Configure the extension in VS Code settings:

```json
{
  "llmCodeAnalyzer.serverUrl": "http://localhost:5000",
  "llmCodeAnalyzer.autoAnalyze": false,
  "llmCodeAnalyzer.preferredModel": "deepseek"
}
```

## Requirements

- VS Code 1.74.0 or higher
- LLM Code Analyzer server running on localhost:5000 (or configured URL)
- Valid API keys for the AI models you want to use

## Development

### Building

```bash
npm run compile
```

### Watching for Changes

```bash
npm run watch
```

### Testing

```bash
npm test
```

## API Integration

This extension communicates with the LLM Code Analyzer web server to perform analysis. Make sure the server is running and accessible at the configured URL.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details 