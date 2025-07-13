# Multi-Modal Analysis Troubleshooting Guide

This guide helps you resolve issues with the multimodal analysis feature using Gemini Vision.

## 🚨 Common Issues and Solutions

### 1. "Cannot upload file" or "No image file uploaded"

**Symptoms:**
- File upload button doesn't work
- Error message: "No image file uploaded"
- File selection dialog doesn't open

**Solutions:**

#### Check Browser Compatibility
```bash
# Ensure you're using a modern browser that supports File API
# Tested browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
```

#### Check File Format
- **Supported formats:** PNG, JPG, JPEG, GIF, BMP, WebP
- **File size limit:** 10MB maximum
- **Common issues:**
  - HEIC files (iPhone photos) - convert to JPEG first
  - PDF files - extract images first
  - Corrupted image files

#### Check Console for Errors
1. Open browser Developer Tools (F12)
2. Go to Console tab
3. Try uploading a file
4. Look for JavaScript errors

### 2. "Gemini API key not found" Error

**Symptoms:**
- Error message: "Gemini API key not found"
- Analysis fails immediately

**Solutions:**

#### Set Up API Key
```bash
# Run the setup script
python setup_api_keys.py

# Or manually create .env file
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

#### Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

#### Verify API Key
```bash
# Test your API key
python test_gemini_multimodal.py
```

### 3. "Multi-modal analysis not available" Error

**Symptoms:**
- Error message: "Multi-modal analysis not available"
- Import errors in console

**Solutions:**

#### Install Dependencies
```bash
# Install required packages
pip install python-dotenv requests Pillow flask

# Or install all requirements
pip install -r requirements.txt
```

#### Check Python Environment
```bash
# Ensure you're in the correct virtual environment
# If using venv:
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Check Python version (3.8+ required)
python --version
```

### 4. "Network error" or "Server error"

**Symptoms:**
- Network connectivity issues
- 500 Internal Server Error
- Timeout errors

**Solutions:**

#### Check Server Status
```bash
# Test if server is running
curl http://localhost:5000/health

# Check server logs for errors
python run_local.py
```

#### Check Network Connectivity
```bash
# Test Gemini API connectivity
python -c "
import requests
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GEMINI_API_KEY')
if key:
    response = requests.get(f'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash?key={key}')
    print(f'Status: {response.status_code}')
else:
    print('No API key found')
"
```

### 5. "File size must be less than 10MB" Error

**Solutions:**

#### Compress Image
```python
# Use PIL to compress image
from PIL import Image
import io

def compress_image(input_path, output_path, max_size_mb=9):
    img = Image.open(input_path)
    
    # Calculate quality to achieve target size
    quality = 95
    while quality > 10:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            break
        quality -= 5
    
    img.save(output_path, format='JPEG', quality=quality)
    print(f"Compressed image saved to {output_path}")
```

#### Use Online Tools
- [TinyPNG](https://tinypng.com/) - Compress PNG/JPEG
- [Squoosh](https://squoosh.app/) - Google's image compression tool

### 6. "Invalid file type" Error

**Solutions:**

#### Convert File Format
```python
# Convert image to supported format
from PIL import Image

def convert_image(input_path, output_path):
    img = Image.open(input_path)
    img = img.convert('RGB')  # Convert to RGB
    img.save(output_path, format='JPEG')
    print(f"Converted image saved to {output_path}")
```

#### Use Command Line Tools
```bash
# Using ImageMagick (if installed)
convert input.heic output.jpg

# Using ffmpeg (if installed)
ffmpeg -i input.heic output.jpg
```

## 🔧 Advanced Troubleshooting

### Debug Mode

Enable debug logging:

```python
# Add to your .env file
FLASK_DEBUG=1
LOG_LEVEL=DEBUG
```

### Test Individual Components

```bash
# Test API key only
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Gemini Key:', '✅' if os.getenv('GEMINI_API_KEY') else '❌')
"

# Test multimodal analyzer
python test_gemini_multimodal.py

# Test Flask endpoints
python -c "
from code_analyzer.web.app import app
with app.test_client() as client:
    response = client.get('/health')
    print('Health check:', response.status_code)
"
```

### Check System Requirements

```bash
# Check Python version
python --version

# Check available memory
# Windows: Task Manager
# Linux/Mac: free -h

# Check disk space
# Windows: dir
# Linux/Mac: df -h
```

## 📋 Step-by-Step Setup

### Complete Setup Process

1. **Clone and Navigate**
   ```bash
   cd llm-code-analyzer
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install python-dotenv requests Pillow flask
   ```

4. **Configure API Keys**
   ```bash
   python setup_api_keys.py
   ```

5. **Test Configuration**
   ```bash
   python test_gemini_multimodal.py
   ```

6. **Start Application**
   ```bash
   python run_local.py
   ```

7. **Access Application**
   - Open http://localhost:5000
   - Go to "Multi-Modal Analysis" tab
   - Upload an image and test

## 🆘 Getting Help

### Before Asking for Help

1. **Run the test script:**
   ```bash
   python test_gemini_multimodal.py
   ```

2. **Check the logs:**
   - Server logs in terminal
   - Browser console (F12)

3. **Verify your setup:**
   - API key is valid
   - Dependencies are installed
   - File format is supported

### Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "No image file uploaded" | File input not working | Check browser compatibility |
| "Gemini API key not found" | Missing API key | Run `python setup_api_keys.py` |
| "Invalid file type" | Unsupported format | Convert to PNG/JPEG |
| "File size must be less than 10MB" | Image too large | Compress image |
| "Network error" | Connectivity issue | Check internet connection |
| "Server error" | Backend issue | Check server logs |

### Support Resources

- **GitHub Issues:** Report bugs and feature requests
- **Documentation:** Check README.md for setup instructions
- **API Documentation:** [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🎯 Quick Fix Checklist

- [ ] API key configured in `.env` file
- [ ] All dependencies installed
- [ ] Using supported image format (PNG/JPEG)
- [ ] Image size under 10MB
- [ ] Modern browser with File API support
- [ ] Server running on http://localhost:5000
- [ ] No firewall blocking localhost:5000 