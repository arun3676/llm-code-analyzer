"""
Security Analyzer for Code Vulnerability Detection
This module provides AI-powered security analysis and vulnerability detection.
"""

import re
import ast
import json
import subprocess
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability found in code."""
    vulnerability_type: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    description: str
    line_number: int
    code_snippet: str
    file_path: str
    cwe_id: Optional[str] = None
    remediation: Optional[str] = None
    confidence: float = 0.0

@dataclass
class SecurityReport:
    """Complete security analysis report."""
    vulnerabilities: List[SecurityVulnerability]
    summary: Dict[str, int]
    risk_score: float
    recommendations: List[str]
    scan_timestamp: str

class SecurityAnalyzer:
    """
    AI-powered security analyzer for detecting vulnerabilities in code.
    """
    
    def __init__(self):
        """Initialize the security analyzer."""
        # Common vulnerability patterns
        self.vulnerability_patterns = {
            'sql_injection': {
                'patterns': [
                    r'execute\s*\(\s*[\'"][^\'"]*\+.*\+[^\'"]*[\'"]',
                    r'cursor\.execute\s*\(\s*[\'"][^\'"]*\+.*\+[^\'"]*[\'"]',
                    r'query\s*=\s*[\'"][^\'"]*\+.*\+[^\'"]*[\'"]',
                ],
                'severity': 'critical',
                'cwe_id': 'CWE-89',
                'description': 'Potential SQL injection vulnerability detected'
            },
            'xss': {
                'patterns': [
                    r'innerHTML\s*=\s*.*\+.*\+',
                    r'document\.write\s*\(\s*.*\+.*\+',
                    r'\.html\s*\(\s*.*\+.*\+',
                ],
                'severity': 'high',
                'cwe_id': 'CWE-79',
                'description': 'Potential Cross-Site Scripting (XSS) vulnerability'
            },
            'command_injection': {
                'patterns': [
                    r'os\.system\s*\(\s*.*\+.*\+',
                    r'subprocess\.call\s*\(\s*.*\+.*\+',
                    r'exec\s*\(\s*.*\+.*\+',
                ],
                'severity': 'critical',
                'cwe_id': 'CWE-78',
                'description': 'Potential command injection vulnerability'
            },
            'path_traversal': {
                'patterns': [
                    r'open\s*\(\s*.*\+.*\+',
                    r'file\s*\(\s*.*\+.*\+',
                    r'Path\s*\(\s*.*\+.*\+',
                ],
                'severity': 'high',
                'cwe_id': 'CWE-22',
                'description': 'Potential path traversal vulnerability'
            },
            'hardcoded_secrets': {
                'patterns': [
                    r'password\s*=\s*[\'"][^\'"]{8,}[\'"]',
                    r'api_key\s*=\s*[\'"][^\'"]{20,}[\'"]',
                    r'secret\s*=\s*[\'"][^\'"]{8,}[\'"]',
                    r'token\s*=\s*[\'"][^\'"]{20,}[\'"]',
                ],
                'severity': 'high',
                'cwe_id': 'CWE-259',
                'description': 'Hardcoded secret detected'
            },
            'weak_crypto': {
                'patterns': [
                    r'hashlib\.md5\s*\(',
                    r'hashlib\.sha1\s*\(',
                    r'cryptography\.hazmat\.primitives\.hashes\.MD5',
                ],
                'severity': 'medium',
                'cwe_id': 'CWE-327',
                'description': 'Weak cryptographic algorithm detected'
            },
            'insecure_random': {
                'patterns': [
                    r'random\.randint\s*\(',
                    r'random\.choice\s*\(',
                    r'Math\.random\s*\(',
                ],
                'severity': 'medium',
                'cwe_id': 'CWE-338',
                'description': 'Insecure random number generation'
            },
            'debug_code': {
                'patterns': [
                    r'console\.log\s*\(',
                    r'print\s*\(\s*[\'"][^\'"]*debug[^\'"]*[\'"]',
                    r'logging\.debug\s*\(',
                ],
                'severity': 'low',
                'cwe_id': 'CWE-489',
                'description': 'Debug code found in production code'
            }
        }
        
        # Language-specific patterns
        self.language_patterns = {
            'python': {
                'eval_injection': {
                    'patterns': [r'eval\s*\(\s*.*\+.*\+'],
                    'severity': 'critical',
                    'cwe_id': 'CWE-95',
                    'description': 'Potential eval injection vulnerability'
                },
                'pickle_insecurity': {
                    'patterns': [r'pickle\.loads\s*\('],
                    'severity': 'high',
                    'cwe_id': 'CWE-502',
                    'description': 'Unsafe pickle deserialization'
                }
            },
            'javascript': {
                'eval_injection': {
                    'patterns': [r'eval\s*\(\s*.*\+.*\+'],
                    'severity': 'critical',
                    'cwe_id': 'CWE-95',
                    'description': 'Potential eval injection vulnerability'
                },
                'innerHTML_xss': {
                    'patterns': [r'\.innerHTML\s*=\s*.*\+.*\+'],
                    'severity': 'high',
                    'cwe_id': 'CWE-79',
                    'description': 'Potential XSS via innerHTML'
                }
            }
        }
    
    def analyze_code_security(self, code: str, language: str = 'python', file_path: str = '') -> SecurityReport:
        """
        Analyze code for security vulnerabilities.
        
        Args:
            code: Source code to analyze
            language: Programming language
            file_path: Path to the file being analyzed
            
        Returns:
            SecurityReport with found vulnerabilities
        """
        vulnerabilities = []
        
        # Analyze with pattern matching
        pattern_vulns = self._pattern_based_analysis(code, language, file_path)
        vulnerabilities.extend(pattern_vulns)
        
        # Analyze with AST (for Python)
        if language == 'python':
            ast_vulns = self._ast_based_analysis(code, file_path)
            vulnerabilities.extend(ast_vulns)
        
        # Analyze with AI (if available)
        ai_vulns = self._ai_based_analysis(code, language, file_path)
        vulnerabilities.extend(ai_vulns)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)
        
        # Generate summary
        summary = self._generate_summary(vulnerabilities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(vulnerabilities)
        
        return SecurityReport(
            vulnerabilities=vulnerabilities,
            summary=summary,
            risk_score=risk_score,
            recommendations=recommendations,
            scan_timestamp=self._get_timestamp()
        )
    
    def _pattern_based_analysis(self, code: str, language: str, file_path: str) -> List[SecurityVulnerability]:
        """Perform pattern-based vulnerability detection."""
        vulnerabilities = []
        lines = code.split('\n')
        
        # Check general patterns
        for vuln_type, vuln_info in self.vulnerability_patterns.items():
            for pattern in vuln_info['patterns']:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        vulnerability = SecurityVulnerability(
                            vulnerability_type=vuln_type,
                            severity=vuln_info['severity'],
                            description=vuln_info['description'],
                            line_number=line_num,
                            code_snippet=line.strip(),
                            file_path=file_path,
                            cwe_id=vuln_info['cwe_id'],
                            confidence=0.8
                        )
                        vulnerabilities.append(vulnerability)
        
        # Check language-specific patterns
        if language in self.language_patterns:
            for vuln_type, vuln_info in self.language_patterns[language].items():
                for pattern in vuln_info['patterns']:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerability = SecurityVulnerability(
                                vulnerability_type=vuln_type,
                                severity=vuln_info['severity'],
                                description=vuln_info['description'],
                                line_number=line_num,
                                code_snippet=line.strip(),
                                file_path=file_path,
                                cwe_id=vuln_info['cwe_id'],
                                confidence=0.9
                            )
                            vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _ast_based_analysis(self, code: str, file_path: str) -> List[SecurityVulnerability]:
        """Perform AST-based vulnerability detection for Python."""
        vulnerabilities = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for eval usage
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == 'eval':
                        vulnerability = SecurityVulnerability(
                            vulnerability_type='eval_injection',
                            severity='critical',
                            description='eval() function usage detected - potential code injection',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            cwe_id='CWE-95',
                            confidence=1.0
                        )
                        vulnerabilities.append(vulnerability)
                
                # Check for exec usage
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == 'exec':
                        vulnerability = SecurityVulnerability(
                            vulnerability_type='exec_injection',
                            severity='critical',
                            description='exec() function usage detected - potential code injection',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            cwe_id='CWE-95',
                            confidence=1.0
                        )
                        vulnerabilities.append(vulnerability)
                
                # Check for pickle usage
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'pickle' and 
                        node.func.attr == 'loads'):
                        vulnerability = SecurityVulnerability(
                            vulnerability_type='pickle_insecurity',
                            severity='high',
                            description='pickle.loads() usage detected - unsafe deserialization',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=ast.unparse(node),
                            file_path=file_path,
                            cwe_id='CWE-502',
                            confidence=0.9
                        )
                        vulnerabilities.append(vulnerability)
        
        except SyntaxError:
            # Code has syntax errors, skip AST analysis
            pass
        
        return vulnerabilities
    
    def _ai_based_analysis(self, code: str, language: str, file_path: str) -> List[SecurityVulnerability]:
        """
        Perform AI-based vulnerability detection.
        This would integrate with an LLM for advanced analysis.
        """
        # Placeholder for AI-based analysis
        # In a real implementation, this would call an LLM API
        # with security-focused prompts
        return []
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate overall risk score based on vulnerabilities."""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            'critical': 10.0,
            'high': 7.0,
            'medium': 4.0,
            'low': 1.0,
            'info': 0.5
        }
        
        total_score = 0.0
        for vuln in vulnerabilities:
            weight = severity_weights.get(vuln.severity, 1.0)
            total_score += weight * vuln.confidence
        
        # Normalize to 0-100 scale
        max_possible_score = len(vulnerabilities) * 10.0
        risk_score = min(100.0, (total_score / max_possible_score) * 100.0)
        
        return risk_score
    
    def _generate_summary(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Generate summary statistics."""
        summary = {
            'total': len(vulnerabilities),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in vulnerabilities:
            summary[vuln.severity] += 1
        
        return summary
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if not vulnerabilities:
            recommendations.append("No security vulnerabilities detected. Keep up the good security practices!")
            return recommendations
        
        # Count vulnerabilities by type
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_types[vuln.vulnerability_type] = vuln_types.get(vuln.vulnerability_type, 0) + 1
        
        # Generate specific recommendations
        if 'sql_injection' in vuln_types:
            recommendations.append("Use parameterized queries or ORM to prevent SQL injection attacks")
        
        if 'xss' in vuln_types:
            recommendations.append("Sanitize user input and use proper output encoding to prevent XSS attacks")
        
        if 'command_injection' in vuln_types:
            recommendations.append("Avoid using os.system() and subprocess.call() with user input. Use subprocess.run() with proper argument lists")
        
        if 'hardcoded_secrets' in vuln_types:
            recommendations.append("Move secrets to environment variables or secure secret management systems")
        
        if 'weak_crypto' in vuln_types:
            recommendations.append("Use strong cryptographic algorithms (SHA-256, bcrypt, etc.) instead of MD5 or SHA-1")
        
        if 'eval_injection' in vuln_types:
            recommendations.append("Avoid using eval() and exec() functions with user input. Use safer alternatives")
        
        # General recommendations
        if len(vulnerabilities) > 5:
            recommendations.append("Consider implementing a security code review process")
        
        if any(v.severity in ['critical', 'high'] for v in vulnerabilities):
            recommendations.append("Address critical and high severity vulnerabilities immediately")
        
        recommendations.append("Consider using automated security scanning tools in your CI/CD pipeline")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def scan_file(self, file_path: str) -> SecurityReport:
        """Scan a single file for security vulnerabilities."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Detect language from file extension
            language = self._detect_language(file_path)
            
            return self.analyze_code_security(code, language, file_path)
        
        except Exception as e:
            # Return error report
            return SecurityReport(
                vulnerabilities=[],
                summary={'error': 1},
                risk_score=0.0,
                recommendations=[f"Error scanning file: {str(e)}"],
                scan_timestamp=self._get_timestamp()
            )
    
    def scan_directory(self, directory_path: str) -> Dict[str, SecurityReport]:
        """Scan all files in a directory for security vulnerabilities."""
        results = {}
        directory = Path(directory_path)
        
        # Common code file extensions
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in code_extensions:
                try:
                    report = self.scan_file(str(file_path))
                    results[str(file_path)] = report
                except Exception as e:
                    print(f"Error scanning {file_path}: {e}")
        
        return results
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        return language_map.get(ext, 'unknown')
    
    def generate_security_report(self, reports: Dict[str, SecurityReport], output_format: str = 'json') -> str:
        """Generate a comprehensive security report."""
        if output_format == 'json':
            return self._generate_json_report(reports)
        elif output_format == 'html':
            return self._generate_html_report(reports)
        else:
            return self._generate_text_report(reports)
    
    def _generate_json_report(self, reports: Dict[str, SecurityReport]) -> str:
        """Generate JSON format security report."""
        report_data = {
            'scan_timestamp': self._get_timestamp(),
            'total_files': len(reports),
            'files': {}
        }
        
        total_vulns = 0
        total_risk_score = 0.0
        
        for file_path, report in reports.items():
            report_data['files'][file_path] = {
                'vulnerabilities': [
                    {
                        'type': v.vulnerability_type,
                        'severity': v.severity,
                        'description': v.description,
                        'line_number': v.line_number,
                        'code_snippet': v.code_snippet,
                        'cwe_id': v.cwe_id,
                        'confidence': v.confidence
                    }
                    for v in report.vulnerabilities
                ],
                'summary': report.summary,
                'risk_score': report.risk_score,
                'recommendations': report.recommendations
            }
            
            total_vulns += len(report.vulnerabilities)
            total_risk_score += report.risk_score
        
        report_data['overall_summary'] = {
            'total_vulnerabilities': total_vulns,
            'average_risk_score': total_risk_score / len(reports) if reports else 0.0
        }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self, reports: Dict[str, SecurityReport]) -> str:
        """Generate HTML format security report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .file-report { margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .vulnerability { margin: 10px 0; padding: 10px; border-left: 4px solid #ff4444; background-color: #fff5f5; }
                .critical { border-left-color: #ff0000; }
                .high { border-left-color: #ff6600; }
                .medium { border-left-color: #ffaa00; }
                .low { border-left-color: #00aa00; }
                .info { border-left-color: #0066ff; }
                .code-snippet { background-color: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Security Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Total files scanned: {file_count}</p>
            </div>
        """.format(
            timestamp=self._get_timestamp(),
            file_count=len(reports)
        )
        
        for file_path, report in reports.items():
            html += f"""
            <div class="file-report">
                <h2>{file_path}</h2>
                <p>Risk Score: {report.risk_score:.1f}/100</p>
                <p>Vulnerabilities: {len(report.vulnerabilities)}</p>
            """
            
            for vuln in report.vulnerabilities:
                html += f"""
                <div class="vulnerability {vuln.severity}">
                    <h3>{vuln.vulnerability_type.title()} ({vuln.severity.upper()})</h3>
                    <p><strong>Description:</strong> {vuln.description}</p>
                    <p><strong>Line:</strong> {vuln.line_number}</p>
                    <p><strong>CWE:</strong> {vuln.cwe_id or 'N/A'}</p>
                    <div class="code-snippet">{vuln.code_snippet}</div>
                </div>
                """
            
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _generate_text_report(self, reports: Dict[str, SecurityReport]) -> str:
        """Generate text format security report."""
        report = f"Security Analysis Report\n"
        report += f"Generated on: {self._get_timestamp()}\n"
        report += f"Total files scanned: {len(reports)}\n\n"
        
        for file_path, file_report in reports.items():
            report += f"File: {file_path}\n"
            report += f"Risk Score: {file_report.risk_score:.1f}/100\n"
            report += f"Vulnerabilities: {len(file_report.vulnerabilities)}\n\n"
            
            for vuln in file_report.vulnerabilities:
                report += f"  [{vuln.severity.upper()}] {vuln.vulnerability_type}\n"
                report += f"    Description: {vuln.description}\n"
                report += f"    Line: {vuln.line_number}\n"
                report += f"    Code: {vuln.code_snippet}\n\n"
            
            report += "\n"
        
        return report 