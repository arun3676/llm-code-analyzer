CODE_ANALYSIS_PROMPT = """Analyze the following code and provide:
1. Code quality score (0-100)
2. Potential bugs or issues
3. Improvement suggestions
4. Generated documentation

Code:
{code}

Provide your analysis in JSON format with the following structure:
{
    "code_quality_score": float,
    "potential_bugs": list[str],
    "improvement_suggestions": list[str],
    "documentation": str
}"""

DOCUMENTATION_PROMPT = """Generate comprehensive documentation for the following code:
1. First, understand the code's purpose
2. Identify key components and their relationships
3. Document function signatures and their purposes
4. Provide usage examples

Code:
{code}

Generate the documentation in markdown format."""