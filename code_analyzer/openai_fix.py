"""
OpenAI client initialization with proxy issue workaround.
This module provides a specialized function to initialize the OpenAI client
without the proxy issues that occur in the main initialization.
"""

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print("Error: Could not import ChatOpenAI from any package")
        raise

import os
from dotenv import load_dotenv
try:
    import openai
except ImportError:
    print("Error: openai package not found. Install it with 'pip install openai'")
    raise

# Load environment variables
load_dotenv()

def initialize_openai_client():
    """
    Initialize the OpenAI client directly, bypassing proxy issues.
    
    Returns:
        ChatOpenAI: Initialized OpenAI chat model or None if initialization fails
    """
    try:
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OpenAI API key not found in environment variables")
            return None
            
        # Create client without proxy settings
        # First try with direct client initialization
        client = openai.OpenAI(api_key=api_key)
        
        # Test if client works
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print(f"OpenAI test successful: {test_response.choices[0].message.content}")
        
        # Now create the LangChain ChatOpenAI wrapper with minimal parameters
        chat_model = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        return chat_model
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None
        
# Example usage
if __name__ == "__main__":
    client = initialize_openai_client()
    if client:
        print("OpenAI client initialized successfully")
    else:
        print("Failed to initialize OpenAI client")