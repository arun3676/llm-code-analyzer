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
    """Parse LLM response into structured format."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback parsing for non-JSON responses
        lines = response.split('\n')
        result = {
            "code_quality_score": 0,
            "potential_bugs": [],
            "improvement_suggestions": [],
            "documentation": ""
        }
        
        current_section = None
        for line in lines:
            if "quality score" in line.lower():
                try:
                    result["code_quality_score"] = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "bugs" in line.lower():
                current_section = "potential_bugs"
            elif "suggestions" in line.lower():
                current_section = "improvement_suggestions"
            elif "documentation" in line.lower():
                current_section = "documentation"
            elif current_section and line.strip():
                if current_section in ["potential_bugs", "improvement_suggestions"]:
                    result[current_section].append(line.strip())
                else:
                    result[current_section] += line + "\n"
                    
        return result