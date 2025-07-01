"""
GitHub Link Analyzer

This module provides functionality to analyze code from GitHub links.
Supports both individual files and repository analysis.
"""

import re
import requests
import base64
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any
import traceback


class GitHubAnalyzer:
    """Analyzer for GitHub links and repositories."""
    
    def __init__(self):
        """Initialize the GitHub analyzer."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLM-Code-Analyzer/1.0'
        })
    
    def parse_github_url(self, url: str) -> Dict[str, str]:
        """
        Parse a GitHub URL to extract owner, repo, and file path.
        
        Args:
            url: GitHub URL (e.g., https://github.com/owner/repo/blob/main/file.py)
            
        Returns:
            Dict with 'owner', 'repo', 'branch', 'file_path' keys
        """
        # Remove trailing slash and normalize
        url = url.rstrip('/')
        
        # Handle different GitHub URL formats
        patterns = [
            # https://github.com/owner/repo/blob/branch/path/to/file
            r'https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)',
            # https://github.com/owner/repo/blob/main/file (default branch)
            r'https?://github\.com/([^/]+)/([^/]+)/blob/(.+)',
            # https://github.com/owner/repo (repository root)
            r'https?://github\.com/([^/]+)/([^/]+)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                if len(match.groups()) == 4:
                    # Full path with branch
                    return {
                        'owner': match.group(1),
                        'repo': match.group(2),
                        'branch': match.group(3),
                        'file_path': match.group(4)
                    }
                elif len(match.groups()) == 3:
                    # File path without explicit branch (assume main)
                    return {
                        'owner': match.group(1),
                        'repo': match.group(2),
                        'branch': 'main',
                        'file_path': match.group(3)
                    }
                elif len(match.groups()) == 2:
                    # Repository root
                    return {
                        'owner': match.group(1),
                        'repo': match.group(2),
                        'branch': 'main',
                        'file_path': None
                    }
        
        raise ValueError(f"Invalid GitHub URL format: {url}")
    
    def get_file_content(self, owner: str, repo: str, branch: str, file_path: str) -> str:
        """
        Fetch file content from GitHub using the raw content API.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        # Use GitHub's raw content API
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        
        try:
            response = self.session.get(raw_url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch file content: {e}")
    
    def get_repo_files(self, owner: str, repo: str, branch: str = 'main', max_files: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of files from a repository (limited to avoid rate limits).
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            max_files: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        
        try:
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            files = []
            for item in data.get('tree', []):
                if item['type'] == 'blob' and self._is_code_file(item['path']):
                    files.append({
                        'path': item['path'],
                        'size': item['size'],
                        'sha': item['sha']
                    })
                    if len(files) >= max_files:
                        break
            
            return files
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch repository files: {e}")
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if a file is likely a code file based on extension."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.clj',
            '.hs', '.ml', '.fs', '.vb', '.sql', '.sh', '.bash', '.zsh', '.fish',
            '.html', '.css', '.scss', '.sass', '.less', '.xml', '.json', '.yaml',
            '.yml', '.toml', '.ini', '.cfg', '.conf', '.md', '.txt', '.r', '.m',
            '.pl', '.pm', '.tcl', '.lua', '.dart', '.elm', '.ex', '.exs', '.erl'
        }
        
        return any(file_path.lower().endswith(ext) for ext in code_extensions)
    
    def analyze_github_link(self, github_url: str, prompt: str = None, model: str = 'deepseek') -> Dict[str, Any]:
        """
        Analyze code from a GitHub link.
        
        Args:
            github_url: GitHub URL to analyze
            prompt: Custom analysis prompt
            model: Model to use for analysis
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Parse the GitHub URL
            parsed = self.parse_github_url(github_url)
            owner = parsed['owner']
            repo = parsed['repo']
            branch = parsed['branch']
            file_path = parsed['file_path']
            
            if file_path:
                # Single file analysis
                return self._analyze_single_file(owner, repo, branch, file_path, prompt, model)
            else:
                # Repository analysis
                return self._analyze_repository(owner, repo, branch, prompt, model)
                
        except Exception as e:
            print(f"Error analyzing GitHub link: {e}")
            traceback.print_exc()
            return {
                'analysis': f"Error analyzing GitHub link: {str(e)}",
                'code_content': '',
                'file_info': {},
                'suggestions': []
            }
    
    def _analyze_single_file(self, owner: str, repo: str, branch: str, file_path: str, 
                           prompt: str, model: str) -> Dict[str, Any]:
        """Analyze a single file from GitHub."""
        try:
            # Fetch file content
            code_content = self.get_file_content(owner, repo, branch, file_path)
            
            # Get file info
            file_info = {
                'owner': owner,
                'repo': repo,
                'branch': branch,
                'file_path': file_path,
                'file_size': len(code_content),
                'lines': len(code_content.splitlines())
            }
            
            # Use the existing analyzer to analyze the code
            from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer
            analyzer = AdvancedCodeAnalyzer()
            
            # Prepare analysis prompt
            if not prompt:
                prompt = f"""Analyze this code from {owner}/{repo}/{file_path} and provide:

1. **Overview**: What does this code do?
2. **Structure**: How is it organized?
3. **Key Functions/Classes**: What are the main components?
4. **Code Quality**: Any observations about style, patterns, or potential issues?
5. **Suggestions**: Any improvements or best practices to consider?

Please provide a comprehensive analysis."""

            # Analyze the code
            analysis_result = analyzer.analyze_code_advanced(code_content, model=model, file_path=file_path)
            code_analysis = analysis_result.code_analysis
            return {
                'analysis': getattr(code_analysis, 'documentation', ''),
                'code_content': code_content,
                'file_info': file_info,
                'suggestions': getattr(code_analysis, 'improvement_suggestions', []),
                'quality_score': getattr(code_analysis, 'code_quality_score', 0)
            }
            
        except Exception as e:
            raise Exception(f"Failed to analyze single file: {e}")
    
    def _analyze_repository(self, owner: str, repo: str, branch: str, 
                          prompt: str, model: str) -> Dict[str, Any]:
        """Analyze a repository (multiple files)."""
        try:
            # Get repository files
            files = self.get_repo_files(owner, repo, branch, max_files=5)
            
            if not files:
                return {
                    'analysis': "No code files found in the repository.",
                    'code_content': '',
                    'file_info': {'owner': owner, 'repo': repo, 'branch': branch},
                    'suggestions': []
                }
            
            # Fetch content of the first few files
            all_content = []
            file_details = []
            
            for file_info in files[:3]:  # Limit to 3 files for analysis
                try:
                    content = self.get_file_content(owner, repo, branch, file_info['path'])
                    all_content.append(f"=== {file_info['path']} ===\n{content}\n")
                    file_details.append({
                        'path': file_info['path'],
                        'size': file_info['size'],
                        'lines': len(content.splitlines())
                    })
                except Exception as e:
                    print(f"Failed to fetch {file_info['path']}: {e}")
                    continue
            
            if not all_content:
                return {
                    'analysis': "Failed to fetch any files from the repository.",
                    'code_content': '',
                    'file_info': {'owner': owner, 'repo': repo, 'branch': branch},
                    'suggestions': []
                }
            
            # Combine all content
            combined_content = "\n".join(all_content)
            
            # Use the existing analyzer
            from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer
            analyzer = AdvancedCodeAnalyzer()
            
            # Prepare analysis prompt
            if not prompt:
                prompt = f"""Analyze this repository ({owner}/{repo}) and provide:

1. **Repository Overview**: What type of project is this?
2. **Code Structure**: How are the files organized?
3. **Main Components**: What are the key files and their purposes?
4. **Technology Stack**: What languages/frameworks are used?
5. **Code Quality**: Overall observations about the codebase
6. **Suggestions**: Any improvements or recommendations?

Please provide a comprehensive analysis of the repository."""

            # Analyze the combined content
            analysis_result = analyzer.analyze_code_advanced(combined_content, model=model)
            code_analysis = analysis_result.code_analysis
            return {
                'analysis': getattr(code_analysis, 'documentation', ''),
                'code_content': combined_content,
                'file_info': {
                    'owner': owner,
                    'repo': repo,
                    'branch': branch,
                    'files_analyzed': file_details,
                    'total_files': len(files)
                },
                'suggestions': getattr(code_analysis, 'improvement_suggestions', []),
                'quality_score': getattr(code_analysis, 'code_quality_score', 0)
            }
            
        except Exception as e:
            raise Exception(f"Failed to analyze repository: {e}") 