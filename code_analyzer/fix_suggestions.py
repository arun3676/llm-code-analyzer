"""
AI-Powered Code Fix Suggestions
Generates interactive fix suggestions with before/after diffs for detected issues.
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CodeFix:
    """Represents a code fix suggestion."""
    issue_type: str
    severity: str
    description: str
    line_number: int
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    tags: List[str]
    related_links: List[str]

@dataclass
class FixSuggestion:
    """Complete fix suggestion with metadata."""
    issue_id: str
    issue_type: str
    severity: str
    title: str
    description: str
    line_number: int
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    tags: List[str]
    related_links: List[str]
    diff: str
    can_auto_apply: bool
    plain_explanation: str = ''
    learn_more_link: str = ''

class FixSuggestionGenerator:
    """Generates AI-powered code fix suggestions."""
    
    def __init__(self, llm_client=None):
        """Initialize the fix suggestion generator."""
        self.llm_client = llm_client
        
        # Common fix patterns for different issue types
        self.fix_patterns = {
            'security': {
                'sql_injection': {
                    'description': 'Use parameterized queries to prevent SQL injection',
                    'tags': ['security', 'sql', 'injection'],
                    'links': ['https://owasp.org/www-community/attacks/SQL_Injection']
                },
                'xss': {
                    'description': 'Sanitize user input to prevent XSS attacks',
                    'tags': ['security', 'xss', 'web'],
                    'links': ['https://owasp.org/www-community/attacks/xss/']
                },
                'hardcoded_secrets': {
                    'description': 'Move secrets to environment variables',
                    'tags': ['security', 'secrets', 'configuration'],
                    'links': ['https://owasp.org/www-project-top-ten/2017/A2_2017-Broken_Authentication']
                }
            },
            'performance': {
                'nested_loops': {
                    'description': 'Optimize nested loops for better performance',
                    'tags': ['performance', 'complexity', 'algorithms'],
                    'links': ['https://en.wikipedia.org/wiki/Time_complexity']
                },
                'inefficient_data_structures': {
                    'description': 'Use appropriate data structures for better performance',
                    'tags': ['performance', 'data-structures'],
                    'links': ['https://en.wikipedia.org/wiki/Data_structure']
                },
                'memory_leak': {
                    'description': 'Fix potential memory leaks',
                    'tags': ['performance', 'memory', 'leak'],
                    'links': ['https://en.wikipedia.org/wiki/Memory_leak']
                }
            },
            'quality': {
                'unused_variables': {
                    'description': 'Remove unused variables to improve code clarity',
                    'tags': ['quality', 'clean-code', 'maintenance'],
                    'links': ['https://en.wikipedia.org/wiki/Code_smell']
                },
                'long_functions': {
                    'description': 'Break down long functions for better maintainability',
                    'tags': ['quality', 'refactoring', 'maintainability'],
                    'links': ['https://en.wikipedia.org/wiki/Code_refactoring']
                },
                'magic_numbers': {
                    'description': 'Replace magic numbers with named constants',
                    'tags': ['quality', 'readability', 'maintainability'],
                    'links': ['https://en.wikipedia.org/wiki/Magic_number_(programming)']
                }
            }
        }
    
    def generate_fix_suggestions(self, 
                                code: str, 
                                issues: List[Dict[str, Any]], 
                                language: str = 'python') -> List[FixSuggestion]:
        """
        Generate fix suggestions for detected issues.
        
        Args:
            code: Source code
            issues: List of detected issues
            language: Programming language
            
        Returns:
            List of fix suggestions
        """
        suggestions = []
        
        for i, issue in enumerate(issues):
            try:
                # Generate AI-powered fix suggestion
                suggestion = self._generate_ai_fix(code, issue, language, i)
                if suggestion:
                    suggestions.append(suggestion)
            except Exception as e:
                print(f"Error generating fix for issue {i}: {e}")
                # Fallback to pattern-based suggestion
                fallback = self._generate_pattern_fix(code, issue, language, i)
                if fallback:
                    suggestions.append(fallback)
        
        return suggestions
    
    def _generate_ai_fix(self, 
                        code: str, 
                        issue: Dict[str, Any], 
                        language: str, 
                        issue_id: int) -> Optional[FixSuggestion]:
        """Generate AI-powered fix suggestion."""
        if not self.llm_client:
            return None
        
        try:
            # Create prompt for the LLM
            prompt = self._create_fix_prompt(code, issue, language)
            
            # Get AI response
            response = self.llm_client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse AI response
            return self._parse_ai_response(content, issue, issue_id)
            
        except Exception as e:
            print(f"AI fix generation failed: {e}")
            return None
    
    def _create_fix_prompt(self, code: str, issue: Dict[str, Any], language: str) -> str:
        """Create prompt for AI fix generation."""
        issue_type = issue.get('type', 'unknown')
        description = issue.get('description', 'Unknown issue')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')
        
        prompt = f"""
You are an expert code reviewer and fixer. Analyze the following code and provide a specific fix for the detected issue.

Language: {language}
Issue Type: {issue_type}
Description: {description}
Line Number: {line_number}
Code Snippet: {code_snippet}

Full Code Context:
```{language}
{code}
```

Please provide a fix in the following JSON format:
{{
    "title": "Brief title of the fix",
    "explanation": "Detailed explanation of why this fix is needed and how it works",
    "original_code": "The problematic code snippet",
    "fixed_code": "The corrected code snippet",
    "confidence": 0.95,
    "tags": ["tag1", "tag2"],
    "related_links": ["https://example.com/doc1", "https://example.com/doc2"],
    "can_auto_apply": true
}}

Focus on:
1. Providing a clear, actionable fix
2. Explaining why the fix is necessary
3. Including relevant documentation links
4. Making the fix as safe as possible to auto-apply

Return only the JSON response, no additional text.
"""
        return prompt
    
    def _parse_ai_response(self, 
                          content: str, 
                          issue: Dict[str, Any], 
                          issue_id: int) -> Optional[FixSuggestion]:
        """Parse AI response into FixSuggestion object."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            # Create diff
            original = data.get('original_code', '')
            fixed = data.get('fixed_code', '')
            diff = self._generate_diff(original, fixed)
            
            plain_expl, learn_link = self._get_plain_explanation(issue.get('type', 'unknown'), issue.get('language', 'python'))
            
            return FixSuggestion(
                issue_id=f"issue_{issue_id}",
                issue_type=issue.get('type', 'unknown'),
                severity=issue.get('severity', 'medium'),
                title=data.get('title', 'Code Fix'),
                description=issue.get('description', 'Unknown issue'),
                line_number=issue.get('line_number', 0),
                original_code=original,
                fixed_code=fixed,
                explanation=data.get('explanation', ''),
                confidence=data.get('confidence', 0.8),
                tags=data.get('tags', []),
                related_links=data.get('related_links', []),
                diff=diff,
                can_auto_apply=data.get('can_auto_apply', False),
                plain_explanation=data.get('plain_explanation', plain_expl),
                learn_more_link=data.get('learn_more_link', learn_link)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing AI response: {e}")
            return None
    
    def _generate_pattern_fix(self, 
                             code: str, 
                             issue: Dict[str, Any], 
                             language: str, 
                             issue_id: int) -> Optional[FixSuggestion]:
        """Generate pattern-based fix suggestion as fallback."""
        issue_type = issue.get('type', 'unknown')
        description = issue.get('description', 'Unknown issue')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')
        
        # Find pattern for this issue type
        pattern_info = None
        for category, patterns in self.fix_patterns.items():
            if issue_type in patterns:
                pattern_info = patterns[issue_type]
                break
        
        if not pattern_info:
            return None
        
        # Generate simple fix based on pattern
        original_code = code_snippet
        fixed_code = self._apply_pattern_fix(original_code, issue_type, language)
        
        if fixed_code == original_code:
            return None
        
        diff = self._generate_diff(original_code, fixed_code)
        
        plain_expl, learn_link = self._get_plain_explanation(issue_type, language)
        
        return FixSuggestion(
            issue_id=f"issue_{issue_id}",
            issue_type=issue_type,
            severity=issue.get('severity', 'medium'),
            title=f"Fix {issue_type.replace('_', ' ').title()}",
            description=description,
            line_number=line_number,
            original_code=original_code,
            fixed_code=fixed_code,
            explanation=pattern_info['description'],
            confidence=0.7,
            tags=pattern_info['tags'],
            related_links=pattern_info['links'],
            diff=diff,
            can_auto_apply=True,
            plain_explanation=plain_expl,
            learn_more_link=learn_link
        )
    
    def _apply_pattern_fix(self, code: str, issue_type: str, language: str) -> str:
        """Apply pattern-based fixes."""
        if language == 'python':
            return self._apply_python_pattern_fix(code, issue_type)
        elif language == 'javascript':
            return self._apply_javascript_pattern_fix(code, issue_type)
        elif language == 'typescript':
            # For now, use JavaScript logic, but can add TypeScript-specific logic here
            return self._apply_javascript_pattern_fix(code, issue_type)
        else:
            return code
    
    def _apply_python_pattern_fix(self, code: str, issue_type: str) -> str:
        """Apply Python-specific pattern fixes."""
        if issue_type == 'unused_variables':
            # Remove unused variable assignments
            return re.sub(r'(\w+)\s*=\s*[^#\n]+#\s*unused', '', code)
        
        elif issue_type == 'magic_numbers':
            # Replace magic numbers with constants
            return re.sub(r'\b(\d{2,})\b', 'CONSTANT_\\1', code)
        
        elif issue_type == 'long_functions':
            # Add function splitting suggestion
            return f"# TODO: Consider splitting this function\n{code}"
        
        return code
    
    def _apply_javascript_pattern_fix(self, code: str, issue_type: str) -> str:
        """Apply JavaScript-specific pattern fixes."""
        if issue_type == 'unused_variables':
            # Remove unused variable declarations
            return re.sub(r'const\s+(\w+)\s*=\s*[^;]+;\s*//\s*unused', '', code)
        
        elif issue_type == 'magic_numbers':
            # Replace magic numbers with constants
            return re.sub(r'\b(\d{2,})\b', 'CONSTANT_\\1', code)
        
        return code
    
    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate a simple diff between original and fixed code."""
        if original == fixed:
            return "No changes needed"
        
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        diff_lines = []
        max_lines = max(len(original_lines), len(fixed_lines))
        
        for i in range(max_lines):
            original_line = original_lines[i] if i < len(original_lines) else ""
            fixed_line = fixed_lines[i] if i < len(fixed_lines) else ""
            
            if original_line != fixed_line:
                diff_lines.append(f"- {original_line}")
                diff_lines.append(f"+ {fixed_line}")
            else:
                diff_lines.append(f"  {original_line}")
        
        return '\n'.join(diff_lines)
    
    def apply_fix(self, code: str, fix: FixSuggestion) -> str:
        """
        Apply a fix suggestion to the code.
        
        Args:
            code: Original code
            fix: Fix suggestion to apply
            
        Returns:
            Updated code with fix applied
        """
        if not fix.can_auto_apply:
            raise ValueError("This fix cannot be auto-applied")
        
        # Simple string replacement for now
        # In a more sophisticated implementation, you'd use AST manipulation
        return code.replace(fix.original_code, fix.fixed_code)
    
    def get_fix_summary(self, suggestions: List[FixSuggestion]) -> Dict[str, Any]:
        """Get a summary of all fix suggestions."""
        if not suggestions:
            return {
                'total_suggestions': 0,
                'auto_applicable': 0,
                'by_severity': {},
                'by_type': {}
            }
        
        summary = {
            'total_suggestions': len(suggestions),
            'auto_applicable': sum(1 for s in suggestions if s.can_auto_apply),
            'by_severity': {},
            'by_type': {}
        }
        
        for suggestion in suggestions:
            # Count by severity
            severity = suggestion.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by type
            issue_type = suggestion.issue_type
            summary['by_type'][issue_type] = summary['by_type'].get(issue_type, 0) + 1
        
        return summary 

    def _get_plain_explanation(self, issue_type: str, language: str) -> Tuple[str, str]:
        # Curated explanations for common issues
        explanations = {
            'sql_injection': ("SQL injection allows attackers to execute arbitrary SQL code. Always use parameterized queries.", "https://owasp.org/www-community/attacks/SQL_Injection"),
            'any_type': ("Using 'any' in TypeScript disables type safety. Prefer explicit types for better reliability.", "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#any"),
            'unsafe_assertion': ("Unsafe type assertions can lead to runtime errors. Use type guards or safer patterns.", "https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#type-assertions"),
            'hardcoded_secrets': ("Hardcoded secrets can be leaked. Use environment variables or secret managers.", "https://owasp.org/www-project-top-ten/2017/A2_2017-Broken_Authentication"),
            'nested_loops': ("Nested loops can cause performance issues (O(n²) or worse). Consider refactoring.", "https://en.wikipedia.org/wiki/Time_complexity"),
            'eval_injection': ("Use of eval can lead to code injection vulnerabilities. Avoid eval when possible.", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/eval"),
        }
        if issue_type in explanations:
            return explanations[issue_type]
        # Fallback
        return (f"This is a {issue_type.replace('_', ' ')} issue. Review and fix as appropriate.", "https://en.wikipedia.org/wiki/Software_bug") 