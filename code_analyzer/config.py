from typing import Dict, Any

DEFAULT_CONFIG = {
    "models": {
        "deepseek": {
            "name": "deepseek-chat",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "claude": {
            "name": "claude-3-haiku-20240307",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "mercury": {
            "name": "mercury-coder",
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "analysis": {
        "min_quality_score": 0,
        "max_quality_score": 100,
        "timeout": 120,
        "max_memory_mb": 4096
    }
}