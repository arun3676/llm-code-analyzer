#!/usr/bin/env python3
"""
Deployment Helper Script for LLM Code Analyzer
This script validates the deployment setup and provides guidance.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - MISSING")
        return False

def check_requirements():
    """Check if all required files exist."""
    print("🔍 Checking deployment requirements...")
    print("=" * 50)
    
    required_files = [
        ("render.yaml", "Deployment configuration"),
        ("requirements.txt", "Python dependencies"),
        ("wsgi.py", "WSGI entry point"),
        ("code_analyzer/web/app.py", "Flask application"),
        ("code_analyzer/__init__.py", "Package initialization"),
        ("README.md", "Project documentation")
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def check_git_status():
    """Check git status and provide guidance."""
    print("\n🔍 Checking Git status...")
    print("=" * 50)
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository found")
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True)
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected:")
                for file in result.stdout.strip().split('\n'):
                    if file:
                        print(f"   - {file}")
                print("\n💡 Consider committing changes before deployment:")
                print("   git add .")
                print("   git commit -m 'Prepare for deployment'")
                print("   git push origin main")
            else:
                print("✅ No uncommitted changes")
        else:
            print("❌ Not in a git repository")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install Git.")
        return False
    
    return True

def check_environment_variables():
    """Check if environment variables are documented."""
    print("\n🔍 Checking environment variables...")
    print("=" * 50)
    
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "DEEPSEEK_API_KEY",
        "MERCURY_API_KEY"
    ]
    
    print("Required environment variables for deployment:")
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set (local)")
        else:
            print(f"⚠️  {var}: Not set (will need to configure in Render)")
    
    print("\n💡 Remember to set these in Render dashboard:")
    for var in required_vars:
        print(f"   - {var}")

def check_dependencies():
    """Check if requirements.txt is valid."""
    print("\n🔍 Checking dependencies...")
    print("=" * 50)
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        
        print(f"✅ Found {len(lines)} dependency lines")
        
        # Check for common issues
        issues = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#') and '==' not in line and '>=' not in line and '<=' not in line:
                if not line.startswith('sqlite3'):  # sqlite3 is built-in
                    issues.append(f"Line {i}: {line} - No version specified")
        
        if issues:
            print("⚠️  Potential issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Dependencies look good")
            
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False
    
    return True

def provide_deployment_steps():
    """Provide step-by-step deployment instructions."""
    print("\n🚀 Deployment Steps")
    print("=" * 50)
    print("1. Commit your changes to GitHub:")
    print("   git add .")
    print("   git commit -m 'Prepare for deployment'")
    print("   git push origin main")
    print()
    print("2. Go to Render Dashboard:")
    print("   https://dashboard.render.com")
    print()
    print("3. Create New Web Service:")
    print("   - Click 'New +' → 'Web Service'")
    print("   - Connect your GitHub repository")
    print("   - Select the llm-code-analyzer repository")
    print()
    print("4. Configure the service:")
    print("   - Name: llm-code-analyzer")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120")
    print()
    print("5. Set Environment Variables:")
    print("   - OPENAI_API_KEY")
    print("   - ANTHROPIC_API_KEY")
    print("   - DEEPSEEK_API_KEY")
    print("   - MERCURY_API_KEY")
    print()
    print("6. Deploy:")
    print("   - Click 'Create Web Service'")
    print("   - Monitor the build process")
    print()
    print("7. Test your application:")
    print("   - Visit the provided URL")
    print("   - Test code analysis functionality")

def main():
    """Main deployment validation function."""
    print("🚀 LLM Code Analyzer - Deployment Validator")
    print("=" * 60)
    
    # Check all requirements
    files_ok = check_requirements()
    git_ok = check_git_status()
    deps_ok = check_dependencies()
    check_environment_variables()
    
    print("\n" + "=" * 60)
    if files_ok and git_ok and deps_ok:
        print("✅ All checks passed! Ready for deployment.")
        provide_deployment_steps()
    else:
        print("❌ Some issues found. Please fix them before deployment.")
        print("\n💡 Common fixes:")
        if not files_ok:
            print("   - Ensure all required files exist")
        if not git_ok:
            print("   - Initialize git repository and commit changes")
        if not deps_ok:
            print("   - Fix issues in requirements.txt")
    
    print("\n📖 For detailed instructions, see DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main() 