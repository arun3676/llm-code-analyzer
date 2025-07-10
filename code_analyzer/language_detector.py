"""
Language and Framework Detection Module
Detects programming languages and frameworks from code and file extensions.
"""

import re
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class LanguageInfo:
    """Information about a detected programming language."""
    name: str
    version: Optional[str] = None
    confidence: float = 1.0
    file_extension: Optional[str] = None
    shebang: Optional[str] = None

@dataclass
class FrameworkInfo:
    """Information about a detected framework."""
    name: str
    version: Optional[str] = None
    confidence: float = 1.0
    type: str = "web"  # web, mobile, desktop, etc.

@dataclass
class DetectionResult:
    """Result of language and framework detection."""
    language: LanguageInfo
    frameworks: List[FrameworkInfo]
    build_tools: List[str]
    dependencies: Dict[str, str]
    confidence: float

class LanguageDetector:
    """Detects programming languages and frameworks from code and files."""
    
    def __init__(self):
        """Initialize the language detector."""
        # Language detection patterns
        self.language_patterns = {
            'python': {
                'extensions': ['.py', '.pyw', '.pyx', '.pyi'],
                'keywords': ['def ', 'class ', 'import ', 'from ', 'if __name__', 'print(', 'return '],
                'shebang': r'^#!.*python',
                'version_patterns': [
                    r'python(\d+\.\d+)',
                    r'python3',
                    r'#!/usr/bin/env python(\d+\.\d+)?'
                ]
            },
            'javascript': {
                'extensions': ['.js', '.jsx', '.mjs'],
                'keywords': ['function ', 'const ', 'let ', 'var ', 'console.log', 'export ', 'import '],
                'shebang': r'^#!.*node',
                'version_patterns': [
                    r'node(\d+\.\d+)',
                    r'#!/usr/bin/env node'
                ]
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'keywords': [
                    'interface ', 'type ', 'enum ', 'namespace ', 'declare ', ': string', ': number', ': boolean', ': any',
                    'as ', 'implements ', 'readonly ', 'public ', 'private ', 'protected ', 'abstract ', 'import type',
                    'export type', 'export interface', 'export enum', 'export namespace', 'export abstract', 'export declare',
                    'import(', 'import type', 'import(', 'import type', 'import(', 'import type', 'import(', 'import type'
                ],
                'version_patterns': [
                    r'typescript(\d+\.\d+)',
                    r'ts(\d+\.\d+)'
                ]
            },
            'java': {
                'extensions': ['.java'],
                'keywords': ['public class', 'private ', 'public ', 'static void main', 'import java.'],
                'version_patterns': [
                    r'java(\d+)',
                    r'jdk(\d+)'
                ]
            },
            'cpp': {
                'extensions': ['.cpp', '.cc', '.cxx', '.hpp', '.h'],
                'keywords': ['#include', 'int main', 'class ', 'namespace ', 'std::', 'cout <<'],
                'version_patterns': [
                    r'c\+\+(\d+)',
                    r'cpp(\d+)'
                ]
            },
            'csharp': {
                'extensions': ['.cs'],
                'keywords': ['using System', 'namespace ', 'public class', 'static void Main', 'Console.WriteLine'],
                'version_patterns': [
                    r'\.net(\d+)',
                    r'c#(\d+)'
                ]
            },
            'go': {
                'extensions': ['.go'],
                'keywords': ['package main', 'import ', 'func main', 'fmt.Println', 'var ', 'const '],
                'version_patterns': [
                    r'go(\d+\.\d+)',
                    r'golang(\d+\.\d+)'
                ]
            },
            'rust': {
                'extensions': ['.rs'],
                'keywords': ['fn main', 'use ', 'let ', 'mut ', 'println!', 'struct ', 'impl '],
                'version_patterns': [
                    r'rust(\d+\.\d+)',
                    r'cargo(\d+\.\d+)'
                ]
            },
            'php': {
                'extensions': ['.php'],
                'keywords': ['<?php', 'function ', 'echo ', 'require_once', 'class ', 'namespace '],
                'version_patterns': [
                    r'php(\d+\.\d+)'
                ]
            },
            'ruby': {
                'extensions': ['.rb'],
                'keywords': ['def ', 'class ', 'require ', 'puts ', 'attr_accessor', 'module '],
                'shebang': r'^#!.*ruby',
                'version_patterns': [
                    r'ruby(\d+\.\d+)'
                ]
            },
            'swift': {
                'extensions': ['.swift'],
                'keywords': ['import ', 'func ', 'class ', 'var ', 'let ', 'print(', 'guard '],
                'version_patterns': [
                    r'swift(\d+\.\d+)'
                ]
            },
            'kotlin': {
                'extensions': ['.kt', '.kts'],
                'keywords': ['fun ', 'val ', 'var ', 'class ', 'import ', 'println(', 'package '],
                'version_patterns': [
                    r'kotlin(\d+\.\d+)'
                ]
            },
            'scala': {
                'extensions': ['.scala'],
                'keywords': ['def ', 'val ', 'var ', 'class ', 'object ', 'import ', 'println('],
                'version_patterns': [
                    r'scala(\d+\.\d+)'
                ]
            }
        }
        
        # Framework detection patterns
        self.framework_patterns = {
            'python': {
                'django': {
                    'keywords': ['from django', 'Django', 'django.contrib', 'django.db'],
                    'files': ['manage.py', 'settings.py', 'urls.py', 'models.py'],
                    'version_patterns': [r'django==(\d+\.\d+\.\d+)', r'Django==(\d+\.\d+\.\d+)']
                },
                'flask': {
                    'keywords': ['from flask', 'Flask', 'app = Flask', '@app.route'],
                    'files': ['app.py', 'flask_app.py'],
                    'version_patterns': [r'flask==(\d+\.\d+\.\d+)', r'Flask==(\d+\.\d+\.\d+)']
                },
                'fastapi': {
                    'keywords': ['from fastapi', 'FastAPI', '@app.get', '@app.post'],
                    'files': ['main.py', 'app.py'],
                    'version_patterns': [r'fastapi==(\d+\.\d+\.\d+)', r'FastAPI==(\d+\.\d+\.\d+)']
                },
                'pandas': {
                    'keywords': ['import pandas', 'pd.DataFrame', 'pd.read_csv'],
                    'version_patterns': [r'pandas==(\d+\.\d+\.\d+)', r'pandas>=(\d+\.\d+\.\d+)']
                },
                'numpy': {
                    'keywords': ['import numpy', 'np.array', 'np.zeros'],
                    'version_patterns': [r'numpy==(\d+\.\d+\.\d+)', r'numpy>=(\d+\.\d+\.\d+)']
                },
                'tensorflow': {
                    'keywords': ['import tensorflow', 'tf.', 'tf.keras'],
                    'version_patterns': [r'tensorflow==(\d+\.\d+\.\d+)', r'tensorflow>=(\d+\.\d+\.\d+)']
                },
                'pytorch': {
                    'keywords': ['import torch', 'torch.', 'nn.Module'],
                    'version_patterns': [r'torch==(\d+\.\d+\.\d+)', r'torch>=(\d+\.\d+\.\d+)']
                }
            },
            'javascript': {
                'react': {
                    'keywords': ['import React', 'ReactDOM', 'useState', 'useEffect', 'JSX'],
                    'files': ['package.json'],
                    'version_patterns': [r'"react":\s*"(\d+\.\d+\.\d+)"', r'react@(\d+\.\d+\.\d+)']
                },
                'vue': {
                    'keywords': ['import Vue', 'new Vue', 'Vue.component', 'v-if', 'v-for'],
                    'files': ['package.json'],
                    'version_patterns': [r'"vue":\s*"(\d+\.\d+\.\d+)"', r'vue@(\d+\.\d+\.\d+)']
                },
                'angular': {
                    'keywords': ['@Component', '@Injectable', 'ngOnInit', 'Angular'],
                    'files': ['package.json', 'angular.json'],
                    'version_patterns': [r'"@angular/core":\s*"(\d+\.\d+\.\d+)"']
                },
                'express': {
                    'keywords': ['const express', 'app.get', 'app.post', 'express()'],
                    'files': ['package.json'],
                    'version_patterns': [r'"express":\s*"(\d+\.\d+\.\d+)"', r'express@(\d+\.\d+\.\d+)']
                },
                'node': {
                    'keywords': ['require(', 'module.exports', 'process.env'],
                    'files': ['package.json'],
                    'version_patterns': [r'"node":\s*"(\d+\.\d+\.\d+)"']
                }
            },
            'java': {
                'spring': {
                    'keywords': ['@SpringBootApplication', '@RestController', '@Service', 'SpringApplication'],
                    'files': ['pom.xml', 'build.gradle'],
                    'version_patterns': [r'<spring-boot.version>(\d+\.\d+\.\d+)</spring-boot.version>']
                },
                'android': {
                    'keywords': ['import android', 'Activity', 'onCreate', 'R.layout'],
                    'files': ['AndroidManifest.xml', 'build.gradle'],
                    'version_patterns': [r'compileSdkVersion\s+(\d+)']
                }
            },
            'cpp': {
                'qt': {
                    'keywords': ['#include <Q', 'QApplication', 'QWidget', 'QMainWindow'],
                    'files': ['CMakeLists.txt', 'Makefile'],
                    'version_patterns': [r'find_package\(Qt(\d+)']
                },
                'boost': {
                    'keywords': ['#include <boost', 'boost::', 'BOOST_'],
                    'files': ['CMakeLists.txt'],
                    'version_patterns': [r'find_package\(Boost\s+(\d+\.\d+)']
                }
            },
            'csharp': {
                'aspnet': {
                    'keywords': ['using Microsoft.AspNetCore', 'Startup', 'ConfigureServices', 'Configure'],
                    'files': ['.csproj'],
                    'version_patterns': [r'<TargetFramework>net(\d+\.\d+)</TargetFramework>']
                },
                'entity': {
                    'keywords': ['using Microsoft.EntityFrameworkCore', 'DbContext', 'DbSet'],
                    'files': ['.csproj'],
                    'version_patterns': [r'<PackageReference Include="Microsoft.EntityFrameworkCore" Version="(\d+\.\d+\.\d+)"']
                }
            }
        }
        
        # Build tool detection patterns
        self.build_tools = {
            'python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', 'poetry.lock'],
            'javascript': ['package.json', 'yarn.lock', 'package-lock.json', 'webpack.config.js'],
            'java': ['pom.xml', 'build.gradle', 'gradle.properties', 'build.xml'],
            'cpp': ['CMakeLists.txt', 'Makefile', 'configure', 'build.sh'],
            'csharp': ['.csproj', '.sln', 'packages.config', 'Directory.Build.props'],
            'go': ['go.mod', 'go.sum', 'Makefile'],
            'rust': ['Cargo.toml', 'Cargo.lock', 'build.rs'],
            'php': ['composer.json', 'composer.lock', 'package.json'],
            'ruby': ['Gemfile', 'Gemfile.lock', 'Rakefile'],
            'swift': ['Package.swift', 'Podfile', 'Cartfile'],
            'kotlin': ['build.gradle.kts', 'gradle.properties', 'settings.gradle.kts'],
            'scala': ['build.sbt', 'project/build.properties', 'build.sc']
        }
    
    def detect_language(self, code: str, file_path: Optional[str] = None) -> LanguageInfo:
        """
        Detect programming language from code and file path.
        
        Args:
            code: Source code
            file_path: Optional file path for extension-based detection
            
        Returns:
            LanguageInfo with detected language details
        """
        # Try file extension first
        if file_path:
            ext = Path(file_path).suffix.lower()
            for lang, patterns in self.language_patterns.items():
                if ext in patterns['extensions']:
                    version = self._extract_version(code, patterns.get('version_patterns', []))
                    return LanguageInfo(
                        name=lang,
                        version=version,
                        confidence=0.9,
                        file_extension=ext
                    )
        
        # Try shebang detection
        shebang_match = re.search(r'^#!([^\n]+)', code)
        if shebang_match:
            shebang = shebang_match.group(1)
            for lang, patterns in self.language_patterns.items():
                if 'shebang' in patterns and re.search(patterns['shebang'], shebang, re.IGNORECASE):
                    version = self._extract_version(shebang, patterns.get('version_patterns', []))
                    return LanguageInfo(
                        name=lang,
                        version=version,
                        confidence=0.95,
                        shebang=shebang
                    )
        
        # Try keyword-based detection (improved for TypeScript)
        scores = {}
        ts_score = 0
        ts_keywords_found = 0
        for lang, patterns in self.language_patterns.items():
            score = 0
            for keyword in patterns['keywords']:
                if keyword in code:
                    score += 1
                    if lang == 'typescript':
                        ts_keywords_found += 1
            
            if score > 0:
                scores[lang] = score / len(patterns['keywords'])
            if lang == 'typescript':
                ts_score = score
        
        # Prefer TypeScript if at least 2 TypeScript patterns are found
        if ts_keywords_found >= 2:
            version = self._extract_version(code, self.language_patterns['typescript'].get('version_patterns', []))
            return LanguageInfo(
                name='typescript',
                version=version,
                confidence=0.95
            )
        
        if scores:
            # Get the language with highest score
            best_lang = max(scores, key=scores.get)
            if scores[best_lang] > 0.3:  # Minimum threshold
                version = self._extract_version(code, self.language_patterns[best_lang].get('version_patterns', []))
                return LanguageInfo(
                    name=best_lang,
                    version=version,
                    confidence=scores[best_lang]
                )
        
        # Default to Python if no clear detection
        return LanguageInfo(name='python', confidence=0.1)
    
    def detect_frameworks(self, code: str, language: str, file_path: Optional[str] = None) -> List[FrameworkInfo]:
        """
        Detect frameworks used in the code.
        
        Args:
            code: Source code
            language: Detected programming language
            file_path: Optional file path for framework detection
            
        Returns:
            List of detected frameworks
        """
        frameworks = []
        
        if language not in self.framework_patterns:
            return frameworks
        
        for framework_name, patterns in self.framework_patterns[language].items():
            confidence = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in code:
                    confidence += 0.3
            
            # Check specific files
            if file_path and 'files' in patterns:
                file_name = Path(file_path).name
                if file_name in patterns['files']:
                    confidence += 0.5
            
            # Extract version
            version = None
            if 'version_patterns' in patterns:
                version = self._extract_version(code, patterns['version_patterns'])
            
            if confidence > 0.3:  # Minimum threshold
                frameworks.append(FrameworkInfo(
                    name=framework_name,
                    version=version,
                    confidence=min(confidence, 1.0)
                ))
        
        return frameworks
    
    def detect_build_tools(self, directory_path: str) -> List[str]:
        """
        Detect build tools from project files.
        
        Args:
            directory_path: Path to project directory
            
        Returns:
            List of detected build tools
        """
        detected_tools = []
        directory = Path(directory_path)
        
        for tool_files in self.build_tools.values():
            for tool_file in tool_files:
                if (directory / tool_file).exists():
                    detected_tools.append(tool_file)
        
        return detected_tools
    
    def detect_dependencies(self, directory_path: str, language: str) -> Dict[str, str]:
        """
        Detect project dependencies.
        
        Args:
            directory_path: Path to project directory
            language: Programming language
            
        Returns:
            Dictionary of dependency names to versions
        """
        dependencies = {}
        directory = Path(directory_path)
        
        if language == 'python':
            # Check requirements.txt
            req_file = directory / 'requirements.txt'
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            match = re.match(r'([a-zA-Z0-9_-]+)==?([a-zA-Z0-9._-]+)?', line)
                            if match:
                                name, version = match.groups()
                                dependencies[name] = version or 'latest'
            
            # Check setup.py
            setup_file = directory / 'setup.py'
            if setup_file.exists():
                with open(setup_file, 'r') as f:
                    content = f.read()
                    install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if install_requires_match:
                        deps_str = install_requires_match.group(1)
                        for dep in re.findall(r'"([^"]+)"', deps_str):
                            match = re.match(r'([a-zA-Z0-9_-]+)==?([a-zA-Z0-9._-]+)?', dep)
                            if match:
                                name, version = match.groups()
                                dependencies[name] = version or 'latest'
        
        elif language == 'javascript':
            # Check package.json
            package_file = directory / 'package.json'
            if package_file.exists():
                with open(package_file, 'r') as f:
                    try:
                        data = json.load(f)
                        deps = data.get('dependencies', {})
                        dev_deps = data.get('devDependencies', {})
                        all_deps = {**deps, **dev_deps}
                        dependencies.update(all_deps)
                    except json.JSONDecodeError:
                        pass
        
        elif language == 'java':
            # Check pom.xml
            pom_file = directory / 'pom.xml'
            if pom_file.exists():
                with open(pom_file, 'r') as f:
                    content = f.read()
                    dependency_matches = re.findall(r'<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>', content)
                    for name, version in dependency_matches:
                        dependencies[name] = version
        
        return dependencies
    
    def detect_all(self, code: str, file_path: Optional[str] = None, directory_path: Optional[str] = None) -> DetectionResult:
        """
        Perform comprehensive language and framework detection.
        
        Args:
            code: Source code
            file_path: Optional file path
            directory_path: Optional project directory path
            
        Returns:
            DetectionResult with all detected information
        """
        # Detect language
        language = self.detect_language(code, file_path)
        
        # Detect frameworks
        frameworks = self.detect_frameworks(code, language.name, file_path)
        
        # Detect build tools and dependencies
        build_tools = []
        dependencies = {}
        
        if directory_path:
            build_tools = self.detect_build_tools(directory_path)
            dependencies = self.detect_dependencies(directory_path, language.name)
        
        # Calculate overall confidence
        confidence = language.confidence
        if frameworks:
            confidence = max(confidence, max(f.confidence for f in frameworks))
        
        return DetectionResult(
            language=language,
            frameworks=frameworks,
            build_tools=build_tools,
            dependencies=dependencies,
            confidence=confidence
        )
    
    def _extract_version(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract version from text using patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(self.language_patterns.keys())
    
    def get_supported_frameworks(self, language: str) -> List[str]:
        """Get list of supported frameworks for a language."""
        return list(self.framework_patterns.get(language, {}).keys()) 