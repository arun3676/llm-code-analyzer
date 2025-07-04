"""
RAG (Retrieval-Augmented Generation) Code Assistant
This module provides intelligent code search and suggestions using vector embeddings.
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken

@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata."""
    content: str
    file_path: str
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    docstring: Optional[str] = None
    imports: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    snippet: CodeSnippet
    relevance_score: float
    context: str
    explanation: str

class RAGCodeAssistant:
    """
    RAG-powered code assistant that can search through codebases
    and provide intelligent code suggestions.
    """
    
    def __init__(self, codebase_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG code assistant.
        
        Args:
            codebase_path: Path to the codebase to index
            embedding_model: Sentence transformer model to use for embeddings
        """
        self.codebase_path = Path(codebase_path)
        self.embedding_model = embedding_model
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="code_snippets",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Language patterns for code detection
        self.language_patterns = {
            'python': r'\.py$',
            'javascript': r'\.(js|jsx|ts|tsx)$',
            'java': r'\.java$',
            'cpp': r'\.(cpp|cc|cxx|h|hpp)$',
            'go': r'\.go$',
            'rust': r'\.rs$',
            'php': r'\.php$',
            'ruby': r'\.rb$',
            'csharp': r'\.cs$',
            'swift': r'\.swift$',
            'kotlin': r'\.kt$',
            'scala': r'\.scala$',
            'html': r'\.(html|htm)$',
            'css': r'\.css$',
            'sql': r'\.sql$',
            'yaml': r'\.(yaml|yml)$',
            'json': r'\.json$',
            'markdown': r'\.(md|markdown)$',
            'dockerfile': r'Dockerfile$',
            'shell': r'\.(sh|bash|zsh)$'
        }
        
        # Tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print("RAG Code Assistant initialized successfully!")
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, file_path, re.IGNORECASE):
                return lang
        return 'unknown'
    
    def extract_code_metadata(self, content: str, language: str) -> Dict:
        """Extract metadata from code content."""
        metadata = {
            'function_name': None,
            'class_name': None,
            'docstring': None,
            'imports': [],
            'dependencies': []
        }
        
        if language == 'python':
            # Extract function names
            function_pattern = r'def\s+(\w+)\s*\('
            functions = re.findall(function_pattern, content)
            if functions:
                metadata['function_name'] = functions[0]
            
            # Extract class names
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, content)
            if classes:
                metadata['class_name'] = classes[0]
            
            # Extract docstrings
            docstring_pattern = r'"""(.*?)"""'
            docstrings = re.findall(docstring_pattern, content, re.DOTALL)
            if docstrings:
                metadata['docstring'] = docstrings[0].strip()
            
            # Extract imports
            import_pattern = r'^(?:from\s+(\w+(?:\.\w+)*)\s+import|import\s+(\w+(?:\.\w+)*))'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            metadata['imports'] = [imp[0] or imp[1] for imp in imports if imp[0] or imp[1]]
        
        elif language == 'javascript':
            # Extract function names
            function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*\(|let\s+(\w+)\s*=\s*\()'
            functions = re.findall(function_pattern, content)
            if functions:
                metadata['function_name'] = next((f for f in functions[0] if f), None)
            
            # Extract class names
            class_pattern = r'class\s+(\w+)'
            classes = re.findall(class_pattern, content)
            if classes:
                metadata['class_name'] = classes[0]
            
            # Extract imports
            import_pattern = r'import\s+(?:.*?from\s+)?[\'"]([^\'"]+)[\'"]'
            imports = re.findall(import_pattern, content)
            metadata['imports'] = imports
        
        return metadata
    
    def chunk_code(self, content: str, max_tokens: int = 1000) -> List[str]:
        """Split code into chunks based on token count and logical boundaries."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(self.tokenizer.encode(line))
            
            # If adding this line would exceed the limit, save current chunk
            if current_tokens + line_tokens > max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def index_codebase(self, force_reindex: bool = False) -> int:
        """
        Index the entire codebase for search.
        
        Args:
            force_reindex: Whether to force reindexing even if already indexed
            
        Returns:
            Number of files indexed
        """
        if not force_reindex and self.collection.count() > 0:
            print(f"Codebase already indexed with {self.collection.count()} snippets")
            return self.collection.count()
        
        print("Starting codebase indexing...")
        print(f"Codebase path: {self.codebase_path}")
        
        # Clear existing data
        self.chroma_client.delete_collection("code_snippets")
        self.collection = self.chroma_client.create_collection(
            name="code_snippets",
            metadata={"hnsw:space": "cosine"}
        )
        
        indexed_files = 0
        total_snippets = 0
        
        # Walk through the codebase
        for file_path in self.codebase_path.rglob('*'):
            if file_path.is_file():
                language = self.detect_language(str(file_path))
                
                # Debug output
                print(f"Found file: {file_path} (language: {language})")
                
                # Skip non-code files and common directories
                if (language == 'unknown' or 
                    any(part.startswith('.') for part in file_path.parts) or
                    any(part in ['node_modules', '__pycache__', '.git', 'venv', 'env'] for part in file_path.parts)):
                    print(f"Skipping file: {file_path} (reason: language={language}, hidden={any(part.startswith('.') for part in file_path.parts)}, excluded_dir={any(part in ['node_modules', '__pycache__', '.git', 'venv', 'env'] for part in file_path.parts)})")
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip empty files
                    if not content.strip():
                        print(f"Skipping empty file: {file_path}")
                        continue
                    
                    print(f"Processing file: {file_path} (size: {len(content)} chars)")
                    
                    # Extract metadata
                    metadata = self.extract_code_metadata(content, language)
                    
                    # Chunk the code
                    chunks = self.chunk_code(content)
                    
                    for i, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        
                        # Create code snippet
                        snippet = CodeSnippet(
                            content=chunk,
                            file_path=str(file_path.relative_to(self.codebase_path)),
                            language=language,
                            function_name=metadata['function_name'],
                            class_name=metadata['class_name'],
                            docstring=metadata['docstring'],
                            imports=metadata['imports'],
                            dependencies=metadata['dependencies'],
                            line_start=i * 100,  # Approximate line numbers
                            line_end=(i + 1) * 100
                        )
                        
                        # Generate embedding
                        embedding = self.embedder.encode(chunk).tolist()
                        
                        # Create unique ID
                        snippet_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()
                        
                        # Store in vector database
                        self.collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                'file_path': snippet.file_path,
                                'language': snippet.language,
                                'function_name': snippet.function_name or '',
                                'class_name': snippet.class_name or '',
                                'docstring': snippet.docstring or '',
                                'imports': ','.join(snippet.imports),
                                'dependencies': ','.join(snippet.dependencies),
                                'line_start': snippet.line_start,
                                'line_end': snippet.line_end
                            }],
                            ids=[snippet_id]
                        )
                        
                        total_snippets += 1
                    
                    indexed_files += 1
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
        
        print(f"Indexing complete! Indexed {indexed_files} files with {total_snippets} snippets")
        return total_snippets
    
    def search_code(self, query: str, top_k: int = 5, language_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar code in the codebase.
        
        Args:
            query: Search query
            top_k: Number of results to return
            language_filter: Optional language filter
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Build where clause for language filter
            where_clause = {}
            if language_filter:
                where_clause["language"] = language_filter
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance
                    
                    # Parse imports and dependencies from comma-separated strings
                    imports = metadata['imports'].split(',') if metadata['imports'] else []
                    dependencies = metadata['dependencies'].split(',') if metadata['dependencies'] else []
                    
                    # Create code snippet
                    snippet = CodeSnippet(
                        content=doc,
                        file_path=metadata['file_path'],
                        language=metadata['language'],
                        function_name=metadata['function_name'] if metadata['function_name'] else None,
                        class_name=metadata['class_name'] if metadata['class_name'] else None,
                        docstring=metadata['docstring'] if metadata['docstring'] else None,
                        imports=imports,
                        dependencies=dependencies,
                        line_start=metadata['line_start'],
                        line_end=metadata['line_end']
                    )
                    
                    # Generate context and explanation
                    context = f"Found in {snippet.file_path} ({snippet.language})"
                    if snippet.function_name:
                        context += f", function: {snippet.function_name}"
                    if snippet.class_name:
                        context += f", class: {snippet.class_name}"
                    
                    explanation = f"This code snippet from {snippet.file_path} is relevant to your query '{query}' with a similarity score of {similarity_score:.3f}."
                    
                    search_results.append(SearchResult(
                        snippet=snippet,
                        relevance_score=similarity_score,
                        context=context,
                        explanation=explanation
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Error in search_code: {e}")
            return []
    
    def get_code_suggestions(self, code: str, language: str, context: str = "") -> List[Dict]:
        """
        Get intelligent code suggestions based on similar code in the codebase.
        
        Args:
            code: Current code
            language: Programming language
            context: Additional context about what the user is trying to do
            
        Returns:
            List of code suggestions with explanations
        """
        # Create a search query from the code and context
        search_query = f"{code} {context}".strip()
        
        # Search for similar code
        results = self.search_code(search_query, top_k=3, language_filter=language)
        
        suggestions = []
        
        for result in results:
            suggestion = {
                'type': 'similar_code',
                'title': f"Similar code in {result.snippet.file_path}",
                'code': result.snippet.content,
                'explanation': result.explanation,
                'relevance_score': result.relevance_score,
                'file_path': result.snippet.file_path,
                'function_name': result.snippet.function_name,
                'class_name': result.snippet.class_name
            }
            
            # Add improvement suggestions
            if result.relevance_score > 0.7:
                suggestion['improvements'] = [
                    "Consider extracting common patterns into shared functions",
                    "Look for opportunities to create reusable components",
                    "Check if there are established coding patterns in the codebase"
                ]
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def get_import_suggestions(self, code: str, language: str) -> List[str]:
        """Get import suggestions based on code analysis."""
        if language == 'python':
            # Extract potential imports from code
            potential_imports = []
            
            # Common patterns that might need imports
            patterns = [
                r'(\w+)\.\w+\(',  # Method calls
                r'(\w+)\(',       # Function calls
                r'(\w+)\.\w+',    # Attribute access
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, code)
                potential_imports.extend(matches)
            
            # Remove built-ins and common words
            built_ins = {'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'True', 'False', 'None'}
            potential_imports = [imp for imp in set(potential_imports) if imp not in built_ins]
            
            return potential_imports[:5]  # Return top 5 suggestions
        
        return []
    
    def get_codebase_stats(self) -> Dict:
        """Get statistics about the indexed codebase."""
        if self.collection.count() == 0:
            return {'error': 'No code indexed yet'}
        
        # Get all metadata
        all_data = self.collection.get()
        
        # Count by language
        language_counts = {}
        function_count = 0
        class_count = 0
        
        for metadata in all_data['metadatas']:
            lang = metadata['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            if metadata['function_name']:
                function_count += 1
            if metadata['class_name']:
                class_count += 1
        
        return {
            'total_snippets': self.collection.count(),
            'languages': language_counts,
            'functions': function_count,
            'classes': class_count,
            'indexed_at': datetime.now().isoformat()
        } 