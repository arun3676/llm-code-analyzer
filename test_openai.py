"""
Test OpenAI API connection directly without using LangChain.
This will help identify if the issue is with OpenAI API access or with LangChain integration.
"""

import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_openai_direct():
    """Test OpenAI API directly without LangChain."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found in environment variables")
        return False
        
    try:
        # Create OpenAI client
        print("Creating OpenAI client...")
        client = openai.OpenAI(api_key=api_key)
        
        # Test API access
        print("Testing API access...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        # Print result
        content = response.choices[0].message.content
        print(f"OpenAI API Test Successful! Response: '{content}'")
        return True
        
    except Exception as e:
        print(f"Error testing OpenAI API: {e}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OpenAI API Direct Test")
    print("=" * 50)
    
    # Unset proxy variables if they exist
    if "HTTP_PROXY" in os.environ:
        print(f"Unsetting HTTP_PROXY: {os.environ['HTTP_PROXY']}")
        os.environ.pop("HTTP_PROXY")
    
    if "HTTPS_PROXY" in os.environ:
        print(f"Unsetting HTTPS_PROXY: {os.environ['HTTPS_PROXY']}")
        os.environ.pop("HTTPS_PROXY")
    
    success = test_openai_direct()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ OpenAI API is working correctly!")
        print("If you're still having issues with LangChain integration, the problem is in the integration.")
    else:
        print("❌ OpenAI API test failed.")
        print("Please check your API key and network connection.")
    print("=" * 50)