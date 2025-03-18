import time
import json
from typing import Any, Dict
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        if hasattr(result, 'execution_time'):
            result.execution_time = execution_time
        return result
    return wrapper

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response into structured format with improved error handling.
    
    Args:
        response: The raw response string from the LLM
        
    Returns:
        Parsed response as a dictionary with default values if parsing fails
    """
    # Default result structure
    result = {
        "code_quality_score": 70,  # Default reasonable score
        "potential_bugs": [],
        "improvement_suggestions": [],
        "documentation": ""
    }
    
    try:
        # Try to extract JSON from the response
        # Find the first { and the last } for more robust JSON extraction
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            
            # Update result with parsed values
            if 'code_quality_score' in parsed and isinstance(parsed['code_quality_score'], (int, float)):
                result['code_quality_score'] = parsed['code_quality_score']
            
            if 'potential_bugs' in parsed and isinstance(parsed['potential_bugs'], list):
                result['potential_bugs'] = parsed['potential_bugs']
            
            if 'improvement_suggestions' in parsed and isinstance(parsed['improvement_suggestions'], list):
                result['improvement_suggestions'] = parsed['improvement_suggestions']
            
            if 'documentation' in parsed and isinstance(parsed['documentation'], str):
                result['documentation'] = parsed['documentation']
                
            return result
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from response. Using fallback parsing.")
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    # Fallback parsing for non-JSON responses
    print("Using line-by-line parsing as fallback...")
    lines = response.split('\n')
    
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        
        if "quality score" in line_lower or "code quality" in line_lower:
            try:
                # Try to extract a number from the line
                for word in line.split():
                    if word.replace('.', '').isdigit():
                        result["code_quality_score"] = float(word)
                        break
            except ValueError:
                pass
        elif "bug" in line_lower or "issue" in line_lower or "error" in line_lower:
            current_section = "potential_bugs"
        elif "suggestion" in line_lower or "improvement" in line_lower or "recommend" in line_lower:
            current_section = "improvement_suggestions"
        elif "documentation" in line_lower:
            current_section = "documentation"
        elif current_section and line.strip():
            # Remove list markers and clean the line
            clean_line = line.strip()
            if clean_line.startswith('-') or clean_line.startswith('*'):
                clean_line = clean_line[1:].strip()
            if clean_line.startswith('#'):
                # Skip markdown headers
                continue
                
            if current_section in ["potential_bugs", "improvement_suggestions"]:
                if clean_line and clean_line not in result[current_section]:
                    result[current_section].append(clean_line)
            elif current_section == "documentation":
                result[current_section] += line + "\n"
                
    return result