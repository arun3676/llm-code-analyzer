"""
Verify that the OpenAI API key is correctly loaded and working.
"""

import os
import openai
from dotenv import load_dotenv

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Check if API key exists in environment
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"Found API key: {api_key[:5]}...{api_key[-4:]}")
else:
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    print("\nMake sure your .env file contains:")
    print("OPENAI_API_KEY=your_actual_openai_api_key")
    exit(1)

# Try to use the API key
print("\nTesting API key with OpenAI API...")
openai.api_key = api_key

try:
    # Test API call with older style
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        max_tokens=10
    )
    
    content = response.choices[0].message.content
    print(f"API call successful! Response: '{content}'")
    print("\n✅ Your OpenAI API key is working correctly!")
    
    print("\nTo use OpenAI in your code, make sure to set the API key like this:")
    print("import openai")
    print("openai.api_key = os.getenv('OPENAI_API_KEY')")
    
except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    print("\n❌ There was a problem with your API key or the API call.")
    print("This could be due to:")
    print("1. Invalid API key format")
    print("2. API key has been revoked")
    print("3. Network connectivity issues")
    print("4. Rate limiting on your OpenAI account")