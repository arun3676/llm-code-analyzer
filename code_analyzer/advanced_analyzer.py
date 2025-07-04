"""
Advanced Code Analyzer - Integrated Analysis System
This module integrates RAG, security, performance, and multimodal analysis capabilities.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

# Import our new modules with error handling
try:
    from .rag_assistant import RAGCodeAssistant
    RAG_AVAILABLE = True
except ImportError:
    print("Warning: RAG assistant not available")
    RAG_AVAILABLE = False

try:
    from .security_analyzer import SecurityAnalyzer, SecurityReport
    SECURITY_AVAILABLE = True
except ImportError:
    print("Warning: Security analyzer not available")
    SECURITY_AVAILABLE = False
    SecurityReport = None

try:
    from .performance_analyzer import PerformanceAnalyzer, PerformanceReport
    PERFORMANCE_AVAILABLE = True
except ImportError:
    print("Warning: Performance analyzer not available")
    PERFORMANCE_AVAILABLE = False
    PerformanceReport = None

try:
    from .multimodal_analyzer import MultimodalAnalyzer, MultimodalAnalysis
    MULTIMODAL_AVAILABLE = True
except ImportError:
    print("Warning: Multimodal analyzer not available")
    MULTIMODAL_AVAILABLE = False
    MultimodalAnalysis = None

from .main import CodeAnalyzer
from .models import CodeAnalysisResult

@dataclass
class AdvancedAnalysisResult:
    """Complete advanced analysis result combining all analysis types."""
    # Basic code analysis
    code_analysis: Optional[CodeAnalysisResult] = None
    
    # Advanced analyses
    rag_suggestions: Optional[List[Dict[str, Any]]] = None
    security_report: Optional[Any] = None  # SecurityReport if available
    performance_report: Optional[Any] = None  # PerformanceReport if available
    multimodal_analysis: Optional[Any] = None  # MultimodalAnalysis if available
    
    # Metadata
    analysis_timestamp: str = ""
    analysis_duration: float = 0.0
    features_used: List[str] = None
    
    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []

@dataclass
class AnalysisConfig:
    """Configuration for advanced analysis."""
    enable_rag: bool = True
    enable_security: bool = True
    enable_performance: bool = True
    enable_multimodal: bool = True
    codebase_path: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_rag_results: int = 5
    security_scan_level: str = 'standard'  # 'basic', 'standard', 'comprehensive'
    performance_analysis_level: str = 'standard'  # 'basic', 'standard', 'comprehensive'

class AdvancedCodeAnalyzer:
    """
    Advanced code analyzer that integrates multiple analysis capabilities.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the advanced code analyzer.
        
        Args:
            config: Configuration for analysis features
        """
        self.config = config or AnalysisConfig()
        
        # Initialize base analyzer
        self.base_analyzer = CodeAnalyzer()
        
        # Initialize advanced features
        self.rag_assistant = None
        self.security_analyzer = None
        self.performance_analyzer = None
        self.multimodal_analyzer = None
        
        self._initialize_advanced_features()
    
    def _initialize_advanced_features(self):
        """Initialize advanced analysis features based on configuration."""
        try:
            # Initialize RAG assistant
            if self.config.enable_rag and self.config.codebase_path and RAG_AVAILABLE:
                print("Initializing RAG Code Assistant...")
                self.rag_assistant = RAGCodeAssistant(
                    codebase_path=self.config.codebase_path
                )
                # Index the codebase
                try:
                    snippet_count = self.rag_assistant.index_codebase()
                    print(f"RAG assistant initialized with {snippet_count} code snippets")
                except Exception as e:
                    print(f"Warning: RAG indexing failed: {e}")
                    self.rag_assistant = None
            
            # Initialize security analyzer
            if self.config.enable_security and SECURITY_AVAILABLE:
                print("Initializing Security Analyzer...")
                self.security_analyzer = SecurityAnalyzer()
            
            # Initialize performance analyzer
            if self.config.enable_performance and PERFORMANCE_AVAILABLE:
                print("Initializing Performance Analyzer...")
                # Pass the LLM client for AI-powered analysis
                llm_client = self.base_analyzer.models.get('deepseek') if hasattr(self.base_analyzer, 'models') else None
                self.performance_analyzer = PerformanceAnalyzer(llm_client=llm_client)
            
            # Initialize multimodal analyzer
            if self.config.enable_multimodal and self.config.openai_api_key and MULTIMODAL_AVAILABLE:
                print("Initializing Multimodal Analyzer...")
                self.multimodal_analyzer = MultimodalAnalyzer(
                    openai_api_key=self.config.openai_api_key
                )
        
        except Exception as e:
            print(f"Warning: Some advanced features failed to initialize: {e}")
    
    def analyze_code_advanced(self, 
                            code: str, 
                            language: str = 'python',
                            file_path: str = '',
                            model: str = 'deepseek') -> AdvancedAnalysisResult:
        """
        Perform comprehensive code analysis using all available features.
        
        Args:
            code: Source code to analyze
            language: Programming language
            file_path: Path to the file being analyzed
            model: LLM model to use for basic analysis
            
        Returns:
            AdvancedAnalysisResult with all analysis results
        """
        start_time = time.time()
        features_used = []
        
        # Basic code analysis
        code_analysis = None
        try:
            code_analysis = self.base_analyzer.analyze_code(code, model=model)
            features_used.append('basic_analysis')
        except Exception as e:
            print(f"Basic analysis failed: {e}")
        
        # RAG analysis
        rag_suggestions = None
        if self.rag_assistant:
            try:
                rag_suggestions = self.rag_assistant.get_code_suggestions(
                    code, language, "Looking for similar patterns and improvements"
                )
                features_used.append('rag_analysis')
            except Exception as e:
                print(f"RAG analysis failed: {e}")
        
        # Security analysis
        security_report = None
        if self.security_analyzer:
            try:
                security_report = self.security_analyzer.analyze_code_security(
                    code, language, file_path
                )
                features_used.append('security_analysis')
            except Exception as e:
                print(f"Security analysis failed: {e}")
        
        # Performance analysis
        performance_report = None
        if self.performance_analyzer:
            try:
                performance_report = self.performance_analyzer.analyze_code_performance(
                    code, language, file_path
                )
                features_used.append('performance_analysis')
            except Exception as e:
                print(f"Performance analysis failed: {e}")
        
        analysis_duration = time.time() - start_time
        
        return AdvancedAnalysisResult(
            code_analysis=code_analysis,
            rag_suggestions=rag_suggestions,
            security_report=security_report,
            performance_report=performance_report,
            multimodal_analysis=None,  # Not applicable for text code
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration=analysis_duration,
            features_used=features_used
        )
    
    def analyze_image(self, image_path: str, analysis_type: str = 'auto') -> AdvancedAnalysisResult:
        """
        Analyze an image (screenshot, UI mockup, diagram) using multimodal capabilities.
        
        Args:
            image_path: Path to the image file
            analysis_type: Type of analysis ('auto', 'code', 'ui', 'diagram')
            
        Returns:
            AdvancedAnalysisResult with multimodal analysis
        """
        start_time = time.time()
        features_used = []
        
        multimodal_analysis = None
        if self.multimodal_analyzer:
            try:
                multimodal_analysis = self.multimodal_analyzer.analyze_image(
                    image_path, analysis_type
                )
                features_used.append('multimodal_analysis')
                
                # If code was extracted, perform additional analysis
                if multimodal_analysis.code_extraction:
                    # Perform security analysis on extracted code
                    if self.security_analyzer:
                        try:
                            security_report = self.security_analyzer.analyze_code_security(
                                multimodal_analysis.code_extraction, 
                                'python',  # Assume Python for extracted code
                                image_path
                            )
                            features_used.append('security_analysis')
                        except Exception as e:
                            print(f"Security analysis on extracted code failed: {e}")
                    
                    # Perform performance analysis on extracted code
                    if self.performance_analyzer:
                        try:
                            performance_report = self.performance_analyzer.analyze_code_performance(
                                multimodal_analysis.code_extraction,
                                'python',
                                image_path
                            )
                            features_used.append('performance_analysis')
                        except Exception as e:
                            print(f"Performance analysis on extracted code failed: {e}")
                
            except Exception as e:
                print(f"Multimodal analysis failed: {e}")
        
        analysis_duration = time.time() - start_time
        
        return AdvancedAnalysisResult(
            code_analysis=None,
            rag_suggestions=None,
            security_report=None,
            performance_report=None,
            multimodal_analysis=multimodal_analysis,
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration=analysis_duration,
            features_used=features_used
        )
    
    def scan_codebase_security(self, codebase_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform security scan on entire codebase.
        
        Args:
            codebase_path: Path to codebase (uses config path if not provided)
            
        Returns:
            Dictionary mapping file paths to security reports
        """
        if not self.security_analyzer:
            return {}
        
        scan_path = codebase_path or self.config.codebase_path
        if not scan_path:
            raise ValueError("No codebase path provided for security scan")
        
        return self.security_analyzer.scan_directory(scan_path)
    
    def get_codebase_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase."""
        if not self.rag_assistant:
            return {'error': 'RAG assistant not available'}
        
        return self.rag_assistant.get_codebase_stats()
    
    def search_similar_code(self, query: str, top_k: int = 5, language_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar code in the codebase.
        
        Args:
            query: Search query
            top_k: Number of results to return
            language_filter: Optional language filter
            
        Returns:
            List of search results
        """
        if not self.rag_assistant:
            return []
        
        results = self.rag_assistant.search_code(query, top_k, language_filter)
        return [asdict(result) for result in results]
    
    def generate_comprehensive_report(self, analysis_result: AdvancedAnalysisResult, 
                                    output_format: str = 'json') -> str:
        """
        Generate a comprehensive report combining all analysis results.
        
        Args:
            analysis_result: Advanced analysis result
            output_format: Output format ('json', 'html', 'text')
            
        Returns:
            Formatted report string
        """
        if output_format == 'json':
            return self._generate_json_report(analysis_result)
        elif output_format == 'html':
            return self._generate_html_report(analysis_result)
        else:
            return self._generate_text_report(analysis_result)
    
    def _generate_json_report(self, analysis_result: AdvancedAnalysisResult) -> str:
        """Generate JSON format comprehensive report."""
        report_data = {
            'analysis_metadata': {
                'timestamp': analysis_result.analysis_timestamp,
                'duration': analysis_result.analysis_duration,
                'features_used': analysis_result.features_used
            },
            'basic_analysis': None,
            'rag_analysis': None,
            'security_analysis': None,
            'performance_analysis': None,
            'multimodal_analysis': None
        }
        
        # Add basic analysis
        if analysis_result.code_analysis:
            report_data['basic_analysis'] = {
                'quality_score': analysis_result.code_analysis.code_quality_score,
                'potential_bugs': analysis_result.code_analysis.potential_bugs,
                'improvement_suggestions': analysis_result.code_analysis.improvement_suggestions,
                'documentation': analysis_result.code_analysis.documentation,
                'execution_time': analysis_result.code_analysis.execution_time
            }
        
        # Add RAG analysis
        if analysis_result.rag_suggestions:
            report_data['rag_analysis'] = analysis_result.rag_suggestions
        
        # Add security analysis
        if analysis_result.security_report:
            report_data['security_analysis'] = {
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
                    for v in analysis_result.security_report.vulnerabilities
                ],
                'summary': analysis_result.security_report.summary,
                'risk_score': analysis_result.security_report.risk_score,
                'recommendations': analysis_result.security_report.recommendations
            }
        
        # Add performance analysis
        if analysis_result.performance_report:
            report_data['performance_analysis'] = {
                'issues': [
                    {
                        'type': i.issue_type,
                        'severity': i.severity,
                        'description': i.description,
                        'line_number': i.line_number,
                        'code_snippet': i.code_snippet,
                        'impact': i.impact,
                        'suggestion': i.suggestion
                    }
                    for i in analysis_result.performance_report.issues
                ],
                'summary': analysis_result.performance_report.summary,
                'overall_score': analysis_result.performance_report.overall_score,
                'recommendations': analysis_result.performance_report.recommendations,
                'complexity_analysis': analysis_result.performance_report.complexity_analysis
            }
        
        # Add multimodal analysis
        if analysis_result.multimodal_analysis:
            report_data['multimodal_analysis'] = {
                'image_type': analysis_result.multimodal_analysis.image_type,
                'detected_elements': [
                    {
                        'type': e.element_type,
                        'bounding_box': e.bounding_box,
                        'content': e.content,
                        'confidence': e.confidence
                    }
                    for e in analysis_result.multimodal_analysis.detected_elements
                ],
                'code_extraction': analysis_result.multimodal_analysis.code_extraction,
                'ui_analysis': analysis_result.multimodal_analysis.ui_analysis,
                'diagram_analysis': analysis_result.multimodal_analysis.diagram_analysis,
                'suggestions': analysis_result.multimodal_analysis.suggestions,
                'confidence_score': analysis_result.multimodal_analysis.confidence_score
            }
        
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self, analysis_result: AdvancedAnalysisResult) -> str:
        """Generate HTML format comprehensive report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Code Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .section { margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .section h2 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }
                .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ff4444; background-color: #fff5f5; }
                .critical { border-left-color: #ff0000; }
                .high { border-left-color: #ff6600; }
                .medium { border-left-color: #ffaa00; }
                .low { border-left-color: #00aa00; }
                .info { border-left-color: #0066ff; }
                .code-snippet { background-color: #f8f8f8; padding: 10px; border-radius: 3px; font-family: monospace; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }
                .suggestion { background-color: #f0f8ff; padding: 10px; margin: 5px 0; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Advanced Code Analysis Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Analysis Duration:</strong> {duration:.2f} seconds</p>
                <p><strong>Features Used:</strong> {features}</p>
            </div>
        """.format(
            timestamp=analysis_result.analysis_timestamp,
            duration=analysis_result.analysis_duration,
            features=', '.join(analysis_result.features_used)
        )
        
        # Basic Analysis Section
        if analysis_result.code_analysis:
            html += """
            <div class="section">
                <h2>Basic Code Analysis</h2>
                <div class="metric">Quality Score: {score}/100</div>
                <div class="metric">Execution Time: {time:.2f}s</div>
            """.format(
                score=analysis_result.code_analysis.code_quality_score,
                time=analysis_result.code_analysis.execution_time
            )
            
            if analysis_result.code_analysis.potential_bugs:
                html += "<h3>Potential Bugs:</h3>"
                for bug in analysis_result.code_analysis.potential_bugs:
                    html += f'<div class="issue medium">{bug}</div>'
            
            if analysis_result.code_analysis.improvement_suggestions:
                html += "<h3>Improvement Suggestions:</h3>"
                for suggestion in analysis_result.code_analysis.improvement_suggestions:
                    html += f'<div class="suggestion">{suggestion}</div>'
            
            html += "</div>"
        
        # Security Analysis Section
        if analysis_result.security_report:
            html += """
            <div class="section">
                <h2>Security Analysis</h2>
                <div class="metric">Risk Score: {score}/100</div>
                <div class="metric">Vulnerabilities: {count}</div>
            """.format(
                score=analysis_result.security_report.risk_score,
                count=len(analysis_result.security_report.vulnerabilities)
            )
            
            for vuln in analysis_result.security_report.vulnerabilities:
                html += f"""
                <div class="issue {vuln.severity}">
                    <h3>{vuln.vulnerability_type.title()} ({vuln.severity.upper()})</h3>
                    <p><strong>Description:</strong> {vuln.description}</p>
                    <p><strong>Line:</strong> {vuln.line_number}</p>
                    <p><strong>CWE:</strong> {vuln.cwe_id or 'N/A'}</p>
                    <div class="code-snippet">{vuln.code_snippet}</div>
                </div>
                """
            
            html += "</div>"
        
        # Performance Analysis Section
        if analysis_result.performance_report:
            html += """
            <div class="section">
                <h2>Performance Analysis</h2>
                <div class="metric">Performance Score: {score}/100</div>
                <div class="metric">Issues Found: {count}</div>
            """.format(
                score=analysis_result.performance_report.overall_score,
                count=len(analysis_result.performance_report.issues)
            )
            
            for issue in analysis_result.performance_report.issues:
                html += f"""
                <div class="issue {issue.severity}">
                    <h3>{issue.issue_type.title()} ({issue.severity.upper()})</h3>
                    <p><strong>Description:</strong> {issue.description}</p>
                    <p><strong>Impact:</strong> {issue.impact}</p>
                    <p><strong>Suggestion:</strong> {issue.suggestion}</p>
                    <div class="code-snippet">{issue.code_snippet}</div>
                </div>
                """
            
            html += "</div>"
        
        # RAG Analysis Section
        if analysis_result.rag_suggestions:
            html += """
            <div class="section">
                <h2>RAG Code Suggestions</h2>
            """
            
            for suggestion in analysis_result.rag_suggestions:
                html += f"""
                <div class="suggestion">
                    <h3>{suggestion.get('title', 'Code Suggestion')}</h3>
                    <p><strong>Type:</strong> {suggestion.get('type', 'N/A')}</p>
                    <p><strong>Explanation:</strong> {suggestion.get('explanation', 'N/A')}</p>
                    <div class="code-snippet">{suggestion.get('code', 'N/A')}</div>
                </div>
                """
            
            html += "</div>"
        
        # Multimodal Analysis Section
        if analysis_result.multimodal_analysis:
            html += """
            <div class="section">
                <h2>Multimodal Analysis</h2>
                <div class="metric">Image Type: {type}</div>
                <div class="metric">Confidence: {conf}%</div>
            """.format(
                type=analysis_result.multimodal_analysis.image_type,
                conf=int(analysis_result.multimodal_analysis.confidence_score * 100)
            )
            
            if analysis_result.multimodal_analysis.code_extraction:
                html += f"""
                <h3>Extracted Code:</h3>
                <div class="code-snippet">{analysis_result.multimodal_analysis.code_extraction}</div>
                """
            
            for suggestion in analysis_result.multimodal_analysis.suggestions:
                html += f'<div class="suggestion">{suggestion}</div>'
            
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _generate_text_report(self, analysis_result: AdvancedAnalysisResult) -> str:
        """Generate text format comprehensive report."""
        report = "=" * 60 + "\n"
        report += "ADVANCED CODE ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Generated: {analysis_result.analysis_timestamp}\n"
        report += f"Analysis Duration: {analysis_result.analysis_duration:.2f} seconds\n"
        report += f"Features Used: {', '.join(analysis_result.features_used)}\n\n"
        
        # Basic Analysis
        if analysis_result.code_analysis:
            report += "BASIC CODE ANALYSIS\n"
            report += "-" * 20 + "\n"
            report += f"Quality Score: {analysis_result.code_analysis.code_quality_score}/100\n"
            report += f"Execution Time: {analysis_result.code_analysis.execution_time:.2f}s\n\n"
            
            if analysis_result.code_analysis.potential_bugs:
                report += "Potential Bugs:\n"
                for bug in analysis_result.code_analysis.potential_bugs:
                    report += f"  • {bug}\n"
                report += "\n"
            
            if analysis_result.code_analysis.improvement_suggestions:
                report += "Improvement Suggestions:\n"
                for suggestion in analysis_result.code_analysis.improvement_suggestions:
                    report += f"  • {suggestion}\n"
                report += "\n"
        
        # Security Analysis
        if analysis_result.security_report:
            report += "SECURITY ANALYSIS\n"
            report += "-" * 18 + "\n"
            report += f"Risk Score: {analysis_result.security_report.risk_score}/100\n"
            report += f"Vulnerabilities: {len(analysis_result.security_report.vulnerabilities)}\n\n"
            
            for vuln in analysis_result.security_report.vulnerabilities:
                report += f"[{vuln.severity.upper()}] {vuln.vulnerability_type}\n"
                report += f"  Description: {vuln.description}\n"
                report += f"  Line: {vuln.line_number}\n"
                report += f"  Code: {vuln.code_snippet}\n\n"
        
        # Performance Analysis
        if analysis_result.performance_report:
            report += "PERFORMANCE ANALYSIS\n"
            report += "-" * 20 + "\n"
            report += f"Performance Score: {analysis_result.performance_report.overall_score}/100\n"
            report += f"Issues Found: {len(analysis_result.performance_report.issues)}\n\n"
            
            for issue in analysis_result.performance_report.issues:
                report += f"[{issue.severity.upper()}] {issue.issue_type}\n"
                report += f"  Description: {issue.description}\n"
                report += f"  Impact: {issue.impact}\n"
                report += f"  Suggestion: {issue.suggestion}\n\n"
        
        # RAG Analysis
        if analysis_result.rag_suggestions:
            report += "RAG CODE SUGGESTIONS\n"
            report += "-" * 20 + "\n"
            
            for suggestion in analysis_result.rag_suggestions:
                report += f"Title: {suggestion.get('title', 'N/A')}\n"
                report += f"Type: {suggestion.get('type', 'N/A')}\n"
                report += f"Explanation: {suggestion.get('explanation', 'N/A')}\n"
                report += f"Code: {suggestion.get('code', 'N/A')}\n\n"
        
        # Multimodal Analysis
        if analysis_result.multimodal_analysis:
            report += "MULTIMODAL ANALYSIS\n"
            report += "-" * 18 + "\n"
            report += f"Image Type: {analysis_result.multimodal_analysis.image_type}\n"
            report += f"Confidence: {analysis_result.multimodal_analysis.confidence_score:.2f}\n\n"
            
            if analysis_result.multimodal_analysis.code_extraction:
                report += "Extracted Code:\n"
                report += analysis_result.multimodal_analysis.code_extraction + "\n\n"
            
            if analysis_result.multimodal_analysis.suggestions:
                report += "Suggestions:\n"
                for suggestion in analysis_result.multimodal_analysis.suggestions:
                    report += f"  • {suggestion}\n"
        
        return report 