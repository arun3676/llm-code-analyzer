#!/usr/bin/env python3
"""
Heroku Deployment Automation Script
This script helps automate the deployment process to Heroku
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"🔄 Running: {command}")
    
    # Handle Heroku CLI path on Windows
    if command.startswith("heroku"):
        # Try different Heroku CLI paths on Windows
        heroku_paths = [
            "heroku",
            r"C:\Program Files\heroku\bin\heroku.cmd",
            r"C:\Program Files\heroku\bin\heroku.exe"
        ]
        
        for heroku_path in heroku_paths:
            try:
                modified_command = command.replace("heroku", heroku_path)
                result = subprocess.run(modified_command, shell=True, check=check, capture_output=True, text=True)
                if result.stdout:
                    print(f"✅ Output: {result.stdout.strip()}")
                return result
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        # If all paths fail, try the original command
        try:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                print(f"✅ Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            if check:
                sys.exit(1)
            return e
    else:
        # For non-heroku commands, run normally
        try:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                print(f"✅ Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            if check:
                sys.exit(1)
            return e

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if Heroku CLI is installed
    result = run_command("heroku --version", check=False)
    if result.returncode != 0:
        print("❌ Heroku CLI not found. Please install it first:")
        print("   Windows: winget install --id=Heroku.HerokuCLI")
        print("   Or visit: https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    # Check if git is available
    result = run_command("git --version", check=False)
    if result.returncode != 0:
        print("❌ Git not found. Please install Git first.")
        return False
    
    # Check if we're in a git repository
    result = run_command("git status", check=False)
    if result.returncode != 0:
        print("❌ Not in a git repository. Please initialize git first:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        return False
    
    print("✅ All prerequisites met!")
    return True

def get_app_name():
    """Get or create Heroku app name."""
    print("\n📝 App Configuration")
    
    # Check if Heroku remote already exists
    result = run_command("git remote -v | grep heroku", check=False)
    if result.returncode == 0:
        # Extract app name from existing remote
        remote_url = result.stdout.strip().split()[1]
        app_name = remote_url.split('/')[-1].replace('.git', '')
        print(f"✅ Found existing Heroku app: {app_name}")
        return app_name
    
    # Ask user for app name
    while True:
        app_name = input("Enter your Heroku app name (must be unique globally): ").strip()
        if app_name:
            # Check if app name is available
            result = run_command(f"heroku apps:info {app_name}", check=False)
            if result.returncode != 0:
                return app_name
            else:
                print(f"❌ App name '{app_name}' already exists. Please choose another name.")
        else:
            print("❌ App name cannot be empty.")

def create_heroku_app(app_name):
    """Create a new Heroku app."""
    print(f"\n🚀 Creating Heroku app: {app_name}")
    
    # Create the app
    result = run_command(f"heroku create {app_name}")
    if result.returncode != 0:
        print(f"❌ Failed to create app: {app_name}")
        return False
    
    print(f"✅ Heroku app created: {app_name}")
    return True

def set_environment_variables():
    """Set environment variables on Heroku."""
    print("\n🔐 Setting environment variables...")
    
    # Get API keys from user
    api_keys = {}
    required_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "DEEPSEEK_API_KEY",
        "MERCURY_API_KEY"
    ]
    
    print("Please enter your API keys:")
    for key in required_keys:
        while True:
            value = input(f"{key}: ").strip()
            if value:
                api_keys[key] = value
                break
            else:
                print(f"❌ {key} is required.")
    
    # Set environment variables
    for key, value in api_keys.items():
        run_command(f"heroku config:set {key}='{value}'")
    
    # Set other environment variables
    run_command("heroku config:set FLASK_ENV=production")
    run_command("heroku config:set PYTHONPATH=/app")
    
    print("✅ Environment variables set!")

def deploy_to_heroku():
    """Deploy the application to Heroku."""
    print("\n🚀 Deploying to Heroku...")
    
    # Check current branch
    result = run_command("git branch --show-current")
    current_branch = result.stdout.strip()
    
    # Add all files
    run_command("git add .")
    
    # Commit changes
    run_command('git commit -m "Deploy to Heroku"')
    
    # Push to Heroku
    print(f"📤 Pushing to Heroku (branch: {current_branch})...")
    run_command(f"git push heroku {current_branch}:main")
    
    print("✅ Deployment completed!")

def scale_app():
    """Scale the app to at least 1 dyno."""
    print("\n⚖️ Scaling app...")
    run_command("heroku ps:scale web=1")
    print("✅ App scaled to 1 dyno!")

def open_app():
    """Open the app in browser."""
    print("\n🌐 Opening app...")
    run_command("heroku open")

def main():
    """Main deployment function."""
    print("🚀 Heroku Deployment Script for LLM Code Analyzer")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Get app name
    app_name = get_app_name()
    
    # Create app if needed
    if not app_name:
        app_name = input("Enter app name: ").strip()
        if not create_heroku_app(app_name):
            return
    
    # Set environment variables
    set_environment_variables()
    
    # Deploy
    deploy_to_heroku()
    
    # Scale
    scale_app()
    
    # Open app
    open_app()
    
    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Your app is available at: https://{app_name}.herokuapp.com")
    print("\n📋 Next steps:")
    print("1. Test the health endpoint: /health")
    print("2. Test code analysis: /")
    print("3. Test dashboard: /dashboard")
    print("4. Monitor logs: heroku logs --tail")

if __name__ == "__main__":
    main() 