# LLM Code Analyzer - Deployment Guide

## Overview
This guide will help you deploy the LLM Code Analyzer application to Render.com, a cloud platform that supports Python web applications.

## Prerequisites

### 1. API Keys Required
You'll need the following API keys for full functionality:

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)
- **DeepSeek API Key**: Get from [DeepSeek Platform](https://platform.deepseek.com/)
- **Mercury API Key**: Get from [Mercury Platform](https://mercury.ai/)

### 2. Render.com Account
- Sign up at [Render.com](https://render.com)
- Connect your GitHub account

## Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure your code is committed to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Verify these files exist in your repository:**
   - `render.yaml` - Deployment configuration
   - `requirements.txt` - Python dependencies
   - `wsgi.py` - WSGI entry point
   - `code_analyzer/web/app.py` - Flask application

### Step 2: Deploy to Render

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" and select "Web Service"

2. **Connect Repository**
   - Connect your GitHub account if not already connected
   - Select your `llm-code-analyzer` repository
   - Choose the branch (usually `main`)

3. **Configure the Service**
   - **Name**: `llm-code-analyzer` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

4. **Set Environment Variables**
   Click "Advanced" and add these environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key
   - `MERCURY_API_KEY`: Your Mercury API key

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Step 3: Monitor Deployment

1. **Check Build Logs**
   - Monitor the build process in the Render dashboard
   - Address any build errors if they occur

2. **Verify Application**
   - Once deployed, visit your application URL
   - Test the main functionality

## Configuration Details

### render.yaml
The deployment configuration includes:
- Python environment setup
- Gunicorn server configuration
- Environment variable definitions
- Port binding for cloud deployment

### Environment Variables
- `PORT`: Automatically set by Render
- `OPENAI_API_KEY`: Required for OpenAI model analysis
- `ANTHROPIC_API_KEY`: Required for Claude model analysis
- `DEEPSEEK_API_KEY`: Required for DeepSeek model analysis
- `MERCURY_API_KEY`: Required for Mercury model analysis

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility
   - Verify file paths and imports

2. **Runtime Errors**
   - Check application logs in Render dashboard
   - Verify all API keys are set correctly
   - Ensure database files are properly handled

3. **Import Errors**
   - Verify all Python packages are in requirements.txt
   - Check for missing dependencies

### Performance Optimization

1. **Worker Configuration**
   - Adjust `--workers` parameter based on your plan
   - Monitor memory usage and adjust accordingly

2. **Timeout Settings**
   - Increase timeout for complex analysis tasks
   - Monitor request processing times

## Security Considerations

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables for all sensitive data
   - Regularly rotate API keys

2. **Application Security**
   - Keep dependencies updated
   - Monitor for security vulnerabilities
   - Implement rate limiting if needed

## Monitoring and Maintenance

1. **Health Checks**
   - The application includes a `/health` endpoint
   - Monitor application uptime and performance

2. **Logs**
   - Access logs through Render dashboard
   - Monitor for errors and performance issues

3. **Updates**
   - Regularly update dependencies
   - Monitor for security patches
   - Test updates in development before deploying

## Support

If you encounter issues:
1. Check the application logs in Render dashboard
2. Verify all environment variables are set correctly
3. Test locally before deploying
4. Check the GitHub repository for known issues

## Next Steps

After successful deployment:
1. Set up custom domain (optional)
2. Configure SSL certificates
3. Set up monitoring and alerts
4. Implement CI/CD pipeline for automated deployments 