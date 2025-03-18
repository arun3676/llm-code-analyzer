"""
Check and fix OpenAI version issues.
This script will check your installed OpenAI version and recommend fixes.
"""

import sys
import subprocess
import os

def check_openai_version():
    """Check the installed version of OpenAI package."""
    try:
        import openai
        print(f"OpenAI version: {openai.__version__}")
        return openai.__version__
    except ImportError:
        print("OpenAI package is not installed.")
        return None
    except AttributeError:
        print("OpenAI package doesn't have a __version__ attribute.")
        # Try to get version through pip
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("openai").version
            print(f"OpenAI version (from pkg_resources): {version}")
            return version
        except:
            print("Could not determine OpenAI version.")
            return "unknown"

def fix_openai_version():
    """Fix the OpenAI package version."""
    print("\n=== FIXING OPENAI VERSION ===")
    
    # First uninstall current version
    print("Uninstalling current OpenAI package...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "openai"])
    
    # Install the specific version known to work
    print("Installing OpenAI v0.28.1 (known to work with common proxy configurations)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "openai==0.28.1"])
    
    print("\nNOTE: This is using an older OpenAI API version that's compatible with proxy settings.")
    print("If you want to use the newest API, you'll need to modify your code to not use proxies.")

def test_with_old_api():
    """Test with old OpenAI API (v0.28.1)."""
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # This is the older API style
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print(f"Test successful! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error testing with old API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OpenAI Version Checker and Fixer")
    print("=" * 50)
    
    version = check_openai_version()
    
    if version and version.startswith("0.28"):
        print("You already have a compatible version of OpenAI (0.28.x).")
        print("Let's test it...")
        if test_with_old_api():
            print("\n✅ Your OpenAI setup is working correctly with version 0.28.x!")
            print("Use the old API style in your code:")
            print("""
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use this API style
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=5
)
content = response.choices[0].message.content
            """)
        else:
            print("\n❌ Your OpenAI setup is not working even with version 0.28.x.")
            print("This might be an issue with your API key or network connection.")
    else:
        print(f"You have OpenAI version {version}, which may not be compatible with proxies.")
        
        answer = input("\nDo you want to install a compatible version (0.28.1)? (y/n): ")
        if answer.lower() == 'y':
            fix_openai_version()
            print("\nTesting with the newly installed version...")
            if test_with_old_api():
                print("\n✅ Your OpenAI setup is now working correctly!")
                print("Use the old API style in your code as shown above.")
            else:
                print("\n❌ Your OpenAI setup is still not working.")
                print("This might be an issue with your API key or network connection.")
        else:
            print("\nKeeping current OpenAI version. You'll need to modify your code to not use proxies.")