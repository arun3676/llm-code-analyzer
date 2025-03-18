CODE_ANALYSIS_PROMPT = """Analyze the following code and provide a detailed assessment.

Code to analyze:
```
{code}
```

Please analyze this code and return your analysis in VALID JSON format only, using the exact structure below:

{{
    "code_quality_score": <a number between 0-100>,
    "potential_bugs": [
        <list of strings describing potential bugs or issues>
    ],
    "improvement_suggestions": [
        <list of strings with improvement suggestions>
    ]
}}

Your response must be valid, parseable JSON with no other text before or after. Do not include explanations, markdown formatting, or any text outside the JSON structure.
"""

DOCUMENTATION_PROMPT = """Generate comprehensive documentation for the following code:

```
{code}
```

Please provide:
1. A brief description of what the code does
2. Detailed explanation of the key components
3. Function signatures and parameter descriptions
4. Usage examples

Generate the documentation in markdown format.
"""