"""
Multi-Modal Code Analysis Module

This module provides functionality to analyze code screenshots, diagrams, and UI mockups
using vision-capable Large Language Models (LLMs) like GPT-4V and Claude 3.5 Sonnet.

Key Features:
- Image upload and preprocessing
- Vision-language model integration
- Code extraction from screenshots
- Diagram and UI analysis
- Multi-model comparison
"""

import os
import base64
import io
import time
from typing import Dict, Any, List, Optional
from PIL import Image
import traceback
import requests

# Import LLM clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not available. Install with: pip install anthropic")

class MultiModalAnalyzer:
    """
    Analyzer for multi-modal content (images + text) using vision-capable LLMs.
    
    Supports:
    - Code screenshots analysis
    - UML diagram interpretation
    - UI mockup review
    - Architecture diagram analysis
    """
    
    def __init__(self):
        """Initialize the multi-modal analyzer with API clients."""
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                # Use OpenAI v1 API for GPT-4V
                self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print("✅ OpenAI client initialized for GPT-4V")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI client: {e}")
        
        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                print("✅ Anthropic client initialized for Claude 3.5 Sonnet")
            except Exception as e:
                print(f"❌ Failed to initialize Anthropic client: {e}")
    
    def _preprocess_image(self, image_file) -> bytes:
        """
        Preprocess uploaded image for optimal analysis.
        
        Args:
            image_file: FileStorage object from Flask
            
        Returns:
            bytes: Processed image data
        """
        try:
            # Read image
            image_data = image_file.read()
            
            # Open with PIL for preprocessing
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary (some models prefer RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (most vision models have size limits)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # Return original data if preprocessing fails
            image_file.seek(0)
            return image_file.read()
    
    def _encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data to base64 string for API requests."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def analyze_with_gemini_vision(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """
        Analyze image using Google Gemini Vision model.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        print(f"[Gemini] GEMINI_API_KEY present: {'YES' if api_key else 'NO'}")
        if api_key:
            print(f"[Gemini] GEMINI_API_KEY starts with: {api_key[:5]}...{api_key[-4:]}")
        if not api_key:
            print("[Gemini] ERROR: GEMINI_API_KEY not found in environment variables!")
            return {'error': 'Gemini API key not found'}
        try:
            base64_image = self._encode_image_base64(image_data)
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}}
                        ]
                    }
                ]
            }
            response = requests.post(url, json=payload, timeout=60)
            data = response.json()
            print("[Gemini] API response:", data)
            if 'candidates' in data and data['candidates']:
                analysis = data['candidates'][0]['content']['parts'][0]['text']
            else:
                print("[Gemini] ERROR or empty response:", data)
                return {'error': data.get('error', 'No response from Gemini Vision')}
            code_extracted = self._extract_code_from_analysis(analysis)
            suggestions = self._generate_suggestions_from_analysis(analysis)
            return {
                'analysis': analysis,
                'code_extracted': code_extracted,
                'suggestions': suggestions,
                'model': 'Gemini Vision'
            }
        except Exception as e:
            print(f"Error in Gemini Vision analysis: {e}")
            traceback.print_exc()
            return {'error': f'Gemini Vision analysis failed: {str(e)}'}
    
    def analyze_with_claude(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """
        Analyze image using Anthropic's Claude 3.5 Sonnet model.
        
        Args:
            image_data: Image bytes
            prompt: Analysis prompt
            
        Returns:
            Dict containing analysis results
        """
        if not self.anthropic_client:
            return {'error': 'Anthropic client not available'}
        
        try:
            # Encode image
            base64_image = self._encode_image_base64(image_data)
            
            # Create vision prompt
            vision_prompt = f"""
{prompt}

Please provide a comprehensive analysis including:
1. What you see in the image (code, diagrams, UI, etc.)
2. Any code that can be extracted
3. Potential issues or improvements
4. Suggestions for better practices

Be specific and detailed in your analysis.
"""
            
            # Call Claude 3.5 Sonnet
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": vision_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            analysis = message.content[0].text
            
            # Extract code if present
            code_extracted = self._extract_code_from_analysis(analysis)
            
            # Generate suggestions
            suggestions = self._generate_suggestions_from_analysis(analysis)
            
            return {
                'analysis': analysis,
                'code_extracted': code_extracted,
                'suggestions': suggestions,
                'model': 'Claude 3.5 Sonnet'
            }
            
        except Exception as e:
            print(f"Error in Claude analysis: {e}")
            traceback.print_exc()
            return {'error': f'Claude analysis failed: {str(e)}'}
    
    def _extract_code_from_analysis(self, analysis: str) -> str:
        """
        Extract code blocks from analysis text.
        
        Args:
            analysis: Analysis text from LLM
            
        Returns:
            Extracted code or empty string
        """
        # Look for code blocks (```code```)
        import re
        code_blocks = re.findall(r'```(?:[\w-]+)?\n(.*?)\n```', analysis, re.DOTALL)
        
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        # Look for inline code or code-like patterns
        code_patterns = re.findall(r'`([^`]+)`', analysis)
        if code_patterns:
            return '\n'.join(code_patterns)
        
        return ""
    
    def _generate_suggestions_from_analysis(self, analysis: str) -> List[str]:
        """
        Generate improvement suggestions from analysis text.
        
        Args:
            analysis: Analysis text from LLM
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Look for suggestion patterns
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['suggest', 'recommend', 'improve', 'better', 'consider']):
                if line and not line.startswith('#'):
                    suggestions.append(line)
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def analyze_image(self, image_file, prompt: str, model: str = 'gemini-vision') -> Dict[str, Any]:
        """
        Analyze uploaded image using specified model.
        """
        try:
            image_data = self._preprocess_image(image_file)
            if model.lower() == 'gemini-vision':
                return self.analyze_with_gemini_vision(image_data, prompt)
            elif model.lower() == 'claude':
                return self.analyze_with_claude(image_data, prompt)
            else:
                return {'error': f'Unknown model: {model}'}
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def analyze_with_all_models(self, image_file, prompt: str) -> Dict[str, Any]:
        """
        Analyze image using all available vision models (Gemini Vision and Claude).
        """
        try:
            image_data = self._preprocess_image(image_file)
            results = {}
            results['gemini-vision'] = self.analyze_with_gemini_vision(image_data, prompt)
            if self.anthropic_client:
                results['claude'] = self.analyze_with_claude(image_data, prompt)
            return results
        except Exception as e:
            print(f"Error in analyze_with_all_models: {e}")
            traceback.print_exc()
            return {'error': str(e)}

    def get_available_models(self) -> List[str]:
        models = []
        models.append('gemini-vision')
        if self.anthropic_client:
            models.append('claude')
        return models 