"""
Performance Analyzer for Code Optimization
This module provides AI-powered performance analysis and optimization suggestions.
"""

import ast
import re
import time
import cProfile
import pstats
import io
import psutil
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

@dataclass
class PerformanceIssue:
    """Represents a performance issue found in code."""
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    line_number: int
    code_snippet: str
    file_path: str
    impact: str
    suggestion: str
    complexity: Optional[str] = None
    estimated_improvement: Optional[str] = None
    ai_optimization: Optional[str] = None  # LLM-generated optimization

@dataclass
class PerformanceReport:
    """Complete performance analysis report."""
    issues: List[PerformanceIssue]
    summary: Dict[str, Any]
    overall_score: float
    recommendations: List[str]
    complexity_analysis: Dict[str, Any]
    scan_timestamp: str
    ai_insights: Optional[List[str]] = None  # LLM-generated insights
    optimization_examples: Optional[List[Dict[str, str]]] = None  # Code examples

class PerformanceAnalyzer:
    """
    AI-powered performance analyzer for detecting optimization opportunities.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the performance analyzer."""
        self.llm_client = llm_client
        
        # Performance anti-patterns
        self.anti_patterns = {
            'nested_loops': {
                'patterns': [
                    r'for\s+.*\s+in\s+.*:\s*\n\s*for\s+.*\s+in\s+.*:',
                    r'while\s+.*:\s*\n\s*for\s+.*\s+in\s+.*:',
                    r'for\s+.*\s+in\s+.*:\s*\n\s*while\s+.*:',
                ],
                'severity': 'high',
                'description': 'Nested loops detected - O(n²) or worse complexity',
                'suggestion': 'Consider using list comprehensions, map(), or vectorized operations'
            },
            'inefficient_list_operations': {
                'patterns': [
                    r'\.append\s*\(\s*\)\s+in\s+loop',
                    r'list\s*\(\s*\)\s*\+\s*list\s*\(\s*\)',
                    r'\.extend\s*\(\s*\[\s*\]\s*\)',
                ],
                'severity': 'medium',
                'description': 'Inefficient list operations detected',
                'suggestion': 'Use list comprehensions or extend() with proper data'
            },
            'string_concatenation': {
                'patterns': [
                    r'str\s*\+\s*str\s+in\s+loop',
                    r'string\s*\+\s*string\s+in\s+loop',
                ],
                'severity': 'medium',
                'description': 'String concatenation in loops detected',
                'suggestion': 'Use join() or f-strings for better performance'
            },
            'unnecessary_computations': {
                'patterns': [
                    r'len\s*\(\s*.*\s*\)\s+in\s+loop',
                    r'range\s*\(\s*len\s*\(\s*.*\s*\)\s*\)',
                ],
                'severity': 'low',
                'description': 'Unnecessary computations in loops',
                'suggestion': 'Cache len() results or use enumerate()'
            },
            'memory_inefficient': {
                'patterns': [
                    r'list\s*\(\s*range\s*\(\s*\d+\s*\)\s*\)',
                    r'\[\s*.*\s+for\s+.*\s+in\s+range\s*\(\s*\d+\s*\)\s*\]',
                ],
                'severity': 'medium',
                'description': 'Memory inefficient operations detected',
                'suggestion': 'Use generators or iterators for large ranges'
            },
            'inefficient_data_structures': {
                'patterns': [
                    r'list\s*\(\s*\)\s+for\s+lookup',
                    r'\.index\s*\(\s*\)\s+on\s+list',
                    r'\.find\s*\(\s*\)\s+on\s+string\s+in\s+loop',
                ],
                'severity': 'high',
                'description': 'Inefficient data structure usage detected',
                'suggestion': 'Use sets for lookups, dictionaries for key-value pairs'
            },
            'recursive_without_memoization': {
                'patterns': [
                    r'def\s+\w+\s*\([^)]*\):\s*\n\s*return\s+\w+\s*\([^)]*\)\s*\+\s*\w+\s*\([^)]*\)',
                    r'def\s+\w+\s*\([^)]*\):\s*\n\s*if\s+.*:\s*\n\s*return\s+\w+\s*\([^)]*\)',
                ],
                'severity': 'medium',
                'description': 'Recursive function without memoization detected',
                'suggestion': 'Consider using @functools.lru_cache or manual memoization'
            }
        }
        
        # Complexity patterns
        self.complexity_patterns = {
            'O(1)': ['constant', 'hash_lookup', 'array_access'],
            'O(log n)': ['binary_search', 'tree_traversal'],
            'O(n)': ['linear_search', 'single_loop', 'list_traversal'],
            'O(n log n)': ['sorting', 'merge_sort', 'quick_sort'],
            'O(n²)': ['nested_loops', 'bubble_sort', 'selection_sort'],
            'O(n³)': ['triple_nested_loops', 'matrix_multiplication'],
            'O(2ⁿ)': ['recursive_fibonacci', 'exponential_growth'],
            'O(n!)': ['factorial', 'permutations']
        }
    
    def analyze_code_performance(self, code: str, language: str = 'python', file_path: str = '') -> PerformanceReport:
        """
        Analyze code for performance issues.
        
        Args:
            code: Source code to analyze
            language: Programming language
            file_path: Path to the file being analyzed
            
        Returns:
            PerformanceReport with found issues
        """
        issues = []
        
        # Pattern-based analysis
        pattern_issues = self._pattern_based_analysis(code, language, file_path)
        issues.extend(pattern_issues)
        
        # AST-based analysis (for Python)
        if language == 'python':
            ast_issues = self._ast_based_analysis(code, file_path)
            issues.extend(ast_issues)
        
        # Complexity analysis
        complexity_analysis = self._analyze_complexity(code, language)
        
        # AI-powered analysis and optimization suggestions
        ai_insights = []
        optimization_examples = []
        if self.llm_client and issues:
            ai_insights, optimization_examples = self._generate_ai_insights(code, issues, language)
            
            # Add AI-generated optimizations to issues
            for i, issue in enumerate(issues):
                if i < len(optimization_examples):
                    issue.ai_optimization = optimization_examples[i].get('optimized_code', '')
        
        # Calculate overall score
        overall_score = self._calculate_performance_score(issues, complexity_analysis)
        
        # Generate summary
        summary = self._generate_summary(issues, complexity_analysis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, complexity_analysis)
        
        return PerformanceReport(
            issues=issues,
            summary=summary,
            overall_score=overall_score,
            recommendations=recommendations,
            complexity_analysis=complexity_analysis,
            scan_timestamp=self._get_timestamp(),
            ai_insights=ai_insights,
            optimization_examples=optimization_examples
        )
    
    def _pattern_based_analysis(self, code: str, language: str, file_path: str) -> List[PerformanceIssue]:
        """Perform pattern-based performance analysis."""
        issues = []
        lines = code.split('\n')
        
        for issue_type, issue_info in self.anti_patterns.items():
            for pattern in issue_info['patterns']:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if this is in a loop context
                        if self._is_in_loop_context(lines, line_num):
                            issue = PerformanceIssue(
                                issue_type=issue_type,
                                severity=issue_info['severity'],
                                description=issue_info['description'],
                                line_number=line_num,
                                code_snippet=line.strip(),
                                file_path=file_path,
                                impact=self._get_impact_description(issue_type),
                                suggestion=issue_info['suggestion']
                            )
                            issues.append(issue)
        
        return issues
    
    def _ast_based_analysis(self, code: str, file_path: str) -> List[PerformanceIssue]:
        """Perform AST-based performance analysis for Python."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for inefficient list comprehensions
                if isinstance(node, ast.ListComp):
                    if self._is_inefficient_list_comp(node):
                        issue = PerformanceIssue(
                            issue_type='inefficient_list_comp',
                            severity='medium',
                            description='Inefficient list comprehension detected',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            impact='Memory usage and execution time',
                            suggestion='Consider using generator expressions or map()'
                        )
                        issues.append(issue)
                
                # Check for nested loops
                elif isinstance(node, ast.For):
                    if self._has_nested_loops(node):
                        issue = PerformanceIssue(
                            issue_type='nested_loops',
                            severity='high',
                            description='Nested loops detected - potential O(n²) complexity',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            impact='Quadratic time complexity',
                            suggestion='Consider using itertools.product() or vectorized operations'
                        )
                        issues.append(issue)
                
                # Check for inefficient string operations
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    if self._is_string_concatenation(node):
                        issue = PerformanceIssue(
                            issue_type='string_concatenation',
                            severity='medium',
                            description='String concatenation detected',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            impact='Memory allocation overhead',
                            suggestion='Use join() or f-strings for better performance'
                        )
                        issues.append(issue)
        
        except SyntaxError:
            # Code has syntax errors, skip AST analysis
            pass
        
        return issues
    
    def _analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity and identify bottlenecks."""
        complexity_analysis = {
            'overall_complexity': 'O(1)',
            'functions': {},
            'loops': [],
            'nested_structures': [],
            'potential_bottlenecks': []
        }
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                complexity_analysis.update(self._analyze_python_complexity(tree))
            except SyntaxError:
                pass
        
        return complexity_analysis
    
    def _analyze_python_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python code complexity using AST."""
        analysis = {
            'functions': {},
            'loops': [],
            'nested_structures': [],
            'potential_bottlenecks': []
        }
        
        for node in ast.walk(tree):
            # Analyze functions
            if isinstance(node, ast.FunctionDef):
                func_complexity = self._analyze_function_complexity(node)
                analysis['functions'][node.name] = func_complexity
            
            # Analyze loops
            elif isinstance(node, (ast.For, ast.While)):
                loop_complexity = self._analyze_loop_complexity(node)
                analysis['loops'].append(loop_complexity)
            
            # Check for nested structures
            elif isinstance(node, ast.If):
                if self._has_deep_nesting(node):
                    analysis['nested_structures'].append({
                        'type': 'if_statement',
                        'depth': self._get_nesting_depth(node),
                        'line': getattr(node, 'lineno', 0)
                    })
        
        return analysis
    
    def _analyze_function_complexity(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze complexity of a function."""
        complexity = {
            'cyclomatic_complexity': 1,  # Base complexity
            'lines_of_code': 0,
            'nested_loops': 0,
            'estimated_complexity': 'O(1)'
        }
        
        # Count lines of code
        complexity['lines_of_code'] = len(ast.unparse(func_node).split('\n'))
        
        # Count nested loops and conditional statements
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                complexity['nested_loops'] += 1
            elif isinstance(node, ast.If):
                complexity['cyclomatic_complexity'] += 1
        
        # Estimate time complexity
        if complexity['nested_loops'] >= 2:
            complexity['estimated_complexity'] = 'O(n²) or worse'
        elif complexity['nested_loops'] == 1:
            complexity['estimated_complexity'] = 'O(n)'
        else:
            complexity['estimated_complexity'] = 'O(1)'
        
        return complexity
    
    def _analyze_loop_complexity(self, loop_node: ast.stmt) -> Dict[str, Any]:
        """Analyze complexity of a loop."""
        return {
            'type': type(loop_node).__name__,
            'line': getattr(loop_node, 'lineno', 0),
            'has_nested_loops': self._has_nested_loops(loop_node),
            'estimated_complexity': 'O(n)' if not self._has_nested_loops(loop_node) else 'O(n²) or worse'
        }
    
    def _has_nested_loops(self, node: ast.stmt) -> bool:
        """Check if a node contains nested loops."""
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)) and child != node:
                return True
        return False
    
    def _has_deep_nesting(self, node: ast.stmt, max_depth: int = 3) -> bool:
        """Check if a node has deep nesting."""
        return self._get_nesting_depth(node) > max_depth
    
    def _get_nesting_depth(self, node: ast.stmt) -> int:
        """Get the nesting depth of a node."""
        depth = 0
        current = node
        
        while hasattr(current, 'parent'):
            if isinstance(current.parent, (ast.If, ast.For, ast.While)):
                depth += 1
            current = current.parent
        
        return depth
    
    def _is_in_loop_context(self, lines: List[str], line_num: int) -> bool:
        """Check if a line is within a loop context."""
        # Simple heuristic: check for indentation and loop keywords
        if line_num <= 1:
            return False
        
        # Check previous lines for loop keywords
        for i in range(line_num - 1, max(0, line_num - 10), -1):
            line = lines[i].strip()
            if line.startswith(('for ', 'while ')):
                return True
            elif line.startswith(('if ', 'elif ', 'else:')):
                continue
            elif line == '' or line.startswith('#'):
                continue
            else:
                break
        
        return False
    
    def _is_inefficient_list_comp(self, node: ast.ListComp) -> bool:
        """Check if a list comprehension is inefficient."""
        # Check for multiple generators or complex expressions
        if len(node.generators) > 1:
            return True
        
        # Check for complex expressions
        if isinstance(node.elt, ast.Call):
            return True
        
        return False
    
    def _is_string_concatenation(self, node: ast.BinOp) -> bool:
        """Check if a binary operation is string concatenation."""
        return (isinstance(node.left, ast.Str) or 
                isinstance(node.right, ast.Str) or
                isinstance(node.left, ast.Name) or
                isinstance(node.right, ast.Name))
    
    def _get_impact_description(self, issue_type: str) -> str:
        """Get impact description for an issue type."""
        impacts = {
            'nested_loops': 'Quadratic or worse time complexity',
            'inefficient_list_operations': 'Memory overhead and slower execution',
            'string_concatenation': 'Memory allocation overhead',
            'unnecessary_computations': 'Redundant CPU usage',
            'memory_inefficient': 'Excessive memory usage'
        }
        return impacts.get(issue_type, 'Performance degradation')
    
    def _calculate_performance_score(self, issues: List[PerformanceIssue], complexity_analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        if not issues:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct points for issues
        severity_penalties = {
            'critical': 20.0,
            'high': 15.0,
            'medium': 10.0,
            'low': 5.0
        }
        
        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 5.0)
            score -= penalty
        
        # Deduct points for complexity
        if complexity_analysis.get('overall_complexity') in ['O(n²)', 'O(n³)', 'O(2ⁿ)', 'O(n!)']:
            score -= 10.0
        
        return max(0.0, score)
    
    def _generate_summary(self, issues: List[PerformanceIssue], complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_issues': len(issues),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'overall_complexity': complexity_analysis.get('overall_complexity', 'Unknown'),
            'functions_analyzed': len(complexity_analysis.get('functions', {})),
            'loops_analyzed': len(complexity_analysis.get('loops', []))
        }
        
        for issue in issues:
            summary[issue.severity] += 1
        
        return summary
    
    def _generate_recommendations(self, issues: List[PerformanceIssue], complexity_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if not issues:
            recommendations.append("No performance issues detected. Your code is well-optimized!")
            return recommendations
        
        # Count issues by type
        issue_types = {}
        for issue in issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        # Generate specific recommendations
        if 'nested_loops' in issue_types:
            recommendations.append("Consider using vectorized operations (NumPy) or itertools for nested loops")
        
        if 'inefficient_list_operations' in issue_types:
            recommendations.append("Use list comprehensions or generators instead of append() in loops")
        
        if 'string_concatenation' in issue_types:
            recommendations.append("Use join() or f-strings instead of string concatenation in loops")
        
        if 'memory_inefficient' in issue_types:
            recommendations.append("Consider using generators or iterators for large data structures")
        
        # Complexity-based recommendations
        if complexity_analysis.get('overall_complexity') in ['O(n²)', 'O(n³)']:
            recommendations.append("Consider optimizing algorithms with O(n²) or worse complexity")
        
        # General recommendations
        if len(issues) > 5:
            recommendations.append("Consider using profiling tools to identify bottlenecks")
        
        if any(issue.severity in ['critical', 'high'] for issue in issues):
            recommendations.append("Address critical and high severity performance issues first")
        
        recommendations.append("Consider using caching for expensive computations")
        recommendations.append("Profile your code with cProfile to identify actual bottlenecks")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def profile_function(self, func: callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function's performance using cProfile.
        
        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary containing profiling results
        """
        try:
            import cProfile
            import pstats
            import io
            
            # Create profiler
            pr = cProfile.Profile()
            
            # Profile the function
            pr.enable()
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            pr.disable()
            
            # Get stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            # Extract key metrics
            stats = pr.getstats()
            total_calls = sum(stat.callcount for stat in stats)
            total_time = sum(stat.totaltime for stat in stats)
            
            return {
                'execution_time': execution_time,
                'total_calls': total_calls,
                'total_time': total_time,
                'profile_stats': s.getvalue(),
                'result': result,
                'success': True
            }
            
        except Exception as e:
            logging.error(f'Profiling failed: {e}')
            return {
                'perf': f'Profiling failed: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def benchmark_alternatives(self, code_versions: List[Tuple[str, callable]], *args, **kwargs) -> Dict[str, Any]:
        """Benchmark different code implementations."""
        results = {}
        
        for name, func in code_versions:
            try:
                profile_result = self.profile_function(func, *args, **kwargs)
                results[name] = {
                    'execution_time': profile_result['execution_time'],
                    'profile_stats': profile_result['profile_stats']
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        # Find the fastest version
        fastest = min(results.items(), key=lambda x: x[1].get('execution_time', float('inf')))
        
        results['fastest'] = fastest[0]
        results['comparison'] = {}
        
        # Compare execution times
        for name, result in results.items():
            if name not in ['fastest', 'comparison'] and 'execution_time' in result:
                fastest_time = fastest[1]['execution_time']
                current_time = result['execution_time']
                
                # Avoid division by zero
                if fastest_time > 0 and current_time > 0:
                    speedup = fastest_time / current_time
                    results['comparison'][name] = f"{speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
                elif fastest_time == 0 and current_time == 0:
                    results['comparison'][name] = "same speed (both 0s)"
                elif fastest_time == 0:
                    results['comparison'][name] = "infinitely faster"
                else:
                    results['comparison'][name] = "infinitely slower"
        
        return results
    
    def _generate_ai_insights(self, code: str, issues: List[PerformanceIssue], language: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Generate AI-powered insights and optimization examples."""
        if not self.llm_client:
            return [], []
        
        try:
            # Create a comprehensive prompt for the LLM
            prompt = f"""
You are an expert performance optimization specialist. Analyze the following code and provide specific optimization suggestions for the identified performance issues.

Code to analyze:
```{language}
{code}
```

Performance issues found:
{chr(10).join([f"- {issue.issue_type}: {issue.description} (Line {issue.line_number})" for issue in issues])}

Please provide:
1. 2-3 high-level insights about the overall performance characteristics
2. Specific optimized code examples for each issue, showing before/after comparisons
3. Estimated performance improvements for each optimization

Format your response as JSON:
{{
    "insights": ["insight1", "insight2", "insight3"],
    "optimizations": [
        {{
            "issue_type": "issue_type",
            "original_code": "original code snippet",
            "optimized_code": "optimized code snippet",
            "explanation": "why this optimization helps",
            "estimated_improvement": "e.g., 50% faster, 30% less memory"
        }}
    ]
}}
"""
            
            response = self.llm_client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                result = json.loads(content)
                insights = result.get('insights', [])
                optimizations = result.get('optimizations', [])
                return insights, optimizations
            except json.JSONDecodeError:
                # Fallback: extract insights from text
                insights = [content[:200] + "..." if len(content) > 200 else content]
                return insights, []
                
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return [], []
    
    def profile_memory_usage(self, func: callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a function."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get memory after execution
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        
        return {
            'execution_time': end_time - start_time,
            'memory_used_bytes': memory_used,
            'memory_used_mb': memory_used / (1024 * 1024),
            'result': result,
            'function_name': func.__name__
        }
    
    def analyze_algorithm_complexity(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Analyze algorithm complexity more thoroughly."""
        complexity_analysis = {
            'time_complexity': 'Unknown',
            'space_complexity': 'Unknown',
            'worst_case': 'Unknown',
            'best_case': 'Unknown',
            'average_case': 'Unknown',
            'complexity_factors': []
        }
        
        if language == 'python':
            try:
                tree = ast.parse(code)
                complexity_analysis.update(self._analyze_python_algorithm_complexity(tree))
            except SyntaxError:
                pass
        
        return complexity_analysis
    
    def _analyze_python_algorithm_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python algorithm complexity using AST."""
        analysis = {
            'time_complexity': 'O(1)',
            'space_complexity': 'O(1)',
            'complexity_factors': []
        }
        
        loop_count = 0
        nested_loop_count = 0
        data_structure_usage = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
                if self._has_nested_loops(node):
                    nested_loop_count += 1
            
            # Analyze data structure usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'extend', 'insert']:
                        data_structure_usage.add('list_operations')
                    elif node.func.attr in ['add', 'remove', 'discard']:
                        data_structure_usage.add('set_operations')
                    elif node.func.attr in ['get', 'setdefault', 'update']:
                        data_structure_usage.add('dict_operations')
        
        # Determine time complexity
        if nested_loop_count > 0:
            analysis['time_complexity'] = 'O(n²) or worse'
            analysis['complexity_factors'].append('nested_loops')
        elif loop_count > 0:
            analysis['time_complexity'] = 'O(n)'
            analysis['complexity_factors'].append('single_loop')
        
        # Determine space complexity
        if 'list_operations' in data_structure_usage:
            analysis['space_complexity'] = 'O(n)'
            analysis['complexity_factors'].append('dynamic_data_structures')
        
        return analysis 