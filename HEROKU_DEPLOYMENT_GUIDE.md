# Heroku Deployment Guide for LLM Code Analyzer

## 🚀 Overview
This guide will help you deploy your LLM Code Analyzer with ALL features on Heroku, including:
- ✅ Multi-model AI analysis (OpenAI, Anthropic, DeepSeek, Mercury)
- ✅ Advanced code analysis with RAG
- ✅ Image analysis capabilities
- ✅ GitHub repository analysis
- ✅ Performance profiling
- ✅ Security analysis
- ✅ Interactive dashboard
- ✅ Mobile-responsive UI

## 📋 Prerequisites

### 1. Heroku Account
- Sign up at [heroku.com](https://heroku.com)
- Install Heroku CLI: `npm install -g heroku`

### 2. API Keys
You'll need these API keys:
- **OpenAI API Key**: [Get from OpenAI](https://platform.openai.com/api-keys)
- **Anthropic API Key**: [Get from Anthropic](https://console.anthropic.com/)
- **DeepSeek API Key**: [Get from DeepSeek](https://platform.deepseek.com/)
- **Mercury API Key**: [Get from Mercury](https://mercury.ai/)

### 3. Git Repository
Ensure your code is in a Git repository (GitHub, GitLab, etc.)

## 🛠️ Deployment Steps

### Step 1: Install Heroku CLI
```bash
# Windows (using PowerShell)
winget install --id=Heroku.HerokuCLI

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Login to Heroku
```bash
heroku login
```

### Step 3: Create Heroku App
```bash
# Navigate to your project directory
cd /c/Users/arunk/.cursor/llm-code-analyzer

# Create a new Heroku app
heroku create your-llm-analyzer-app

# Or use a specific name (must be unique globally)
heroku create llm-code-analyzer-2024
```

### Step 4: Set Environment Variables
```bash
# Set all your API keys
heroku config:set OPENAI_API_KEY="your_openai_api_key_here"
heroku config:set ANTHROPIC_API_KEY="your_anthropic_api_key_here"
heroku config:set DEEPSEEK_API_KEY="your_deepseek_api_key_here"
heroku config:set MERCURY_API_KEY="your_mercury_api_key_here"

# Set Flask environment
heroku config:set FLASK_ENV="production"
heroku config:set PYTHONPATH="/app"
```

### Step 5: Add Buildpacks (if needed)
```bash
# Add Python buildpack
heroku buildpacks:set heroku/python
```

### Step 6: Deploy to Heroku
```bash
# Add all files to git (if not already done)
git add .
git commit -m "Prepare for Heroku deployment"

# Deploy to Heroku
git push heroku main
# or if your branch is called master:
git push heroku master
```

### Step 7: Scale the App
```bash
# Scale to at least 1 dyno (required for web apps)
heroku ps:scale web=1
```

### Step 8: Open Your App
```bash
# Open the app in your browser
heroku open
```

## 🔧 Configuration Details

### Dyno Types and Pricing
- **Eco Dyno**: $5/month (512MB RAM, 0.5 CPU)
- **Basic Dyno**: $7/month (512MB RAM, 1 CPU)
- **Standard Dyno**: $25/month (512MB RAM, 1 CPU, better performance)

### Memory Optimization
Your app uses several memory-intensive libraries:
- **ChromaDB**: ~100-200MB
- **Sentence Transformers**: ~200-300MB
- **Pandas/NumPy**: ~50-100MB
- **Total estimated**: ~500-800MB

**Recommendation**: Start with Basic Dyno ($7/month) for better performance.

### Timeout Settings
- **Gunicorn timeout**: 300 seconds (for long AI analysis)
- **Request timeout**: Handled by Heroku's 30-second limit
- **Dyno sleep**: Eco dynos sleep after 30 minutes of inactivity

## 🚨 Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Check build logs
heroku logs --tail

# Common fixes:
# - Ensure all dependencies are in requirements.txt
# - Check Python version compatibility
# - Verify file paths and imports
```

#### 2. Memory Issues
```bash
# Monitor memory usage
heroku logs --tail | grep "Memory"

# If you see memory errors, upgrade to Basic dyno:
heroku ps:type basic
```

#### 3. API Key Issues
```bash
# Verify environment variables
heroku config

# Re-set if needed
heroku config:set OPENAI_API_KEY="your_key"
```

#### 4. Import Errors
```bash
# Check if all files are committed
git status

# Ensure __init__.py files exist in all directories
find . -name "__init__.py" -type f
```

### Performance Optimization

#### 1. Enable Preloading
The Procfile already includes `--preload` for better performance.

#### 2. Optimize Worker Count
```bash
# For Basic dyno (1 CPU), use 2 workers
# For Standard dyno (1 CPU), use 2-3 workers
```

#### 3. Database Optimization
```bash
# Add PostgreSQL for better data persistence
heroku addons:create heroku-postgresql:mini
```

## 📊 Monitoring

### View Logs
```bash
# Real-time logs
heroku logs --tail

# Recent logs
heroku logs --num 100
```

### Monitor Performance
```bash
# Check dyno status
heroku ps

# Monitor resource usage
heroku logs --tail | grep "Memory\|CPU"
```

### Health Check
Visit: `https://your-app-name.herokuapp.com/health`

## 🔄 Updates and Maintenance

### Deploy Updates
```bash
# Make your changes
git add .
git commit -m "Update description"

# Deploy to Heroku
git push heroku main
```

### Restart App
```bash
# Restart all dynos
heroku restart

# Restart specific dyno
heroku restart web.1
```

### Scale Up/Down
```bash
# Scale up for more traffic
heroku ps:scale web=2

# Scale down to save costs
heroku ps:scale web=1
```

## 💰 Cost Estimation

### Monthly Costs
- **Eco Dyno**: $5/month
- **Basic Dyno**: $7/month (recommended)
- **Standard Dyno**: $25/month
- **PostgreSQL Mini**: $5/month (optional)

**Total estimated**: $7-12/month for full functionality

## ✅ Feature Verification

After deployment, test these features:

1. **Basic Code Analysis**: `/` - Upload and analyze code
2. **Multi-Model Analysis**: Test all AI models
3. **Image Analysis**: `/analyze_image` - Upload images
4. **GitHub Analysis**: `/analyze_github` - Analyze repositories
5. **Dashboard**: `/dashboard` - View analytics
6. **Health Check**: `/health` - Verify app status

## 🆘 Support

If you encounter issues:
1. Check the logs: `heroku logs --tail`
2. Verify environment variables: `heroku config`
3. Test locally first: `python run_local.py`
4. Check Heroku status: [status.heroku.com](https://status.heroku.com)

## 🎉 Success!

Your LLM Code Analyzer is now deployed with all features on Heroku! 

**Your app URL**: `https://your-app-name.herokuapp.com`

Share it with others and start analyzing code with AI! 🚀 