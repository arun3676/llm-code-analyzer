from typing import Dict, Any

DEFAULT_CONFIG = {
    "models": {
        "gpt": {
            "name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "claude": {
            "name": "claude-3-sonnet-20240229",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "analysis": {
        "min_quality_score": 0,
        "max_quality_score": 100,
        "timeout": 30
    }
}