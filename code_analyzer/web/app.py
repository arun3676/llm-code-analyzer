import streamlit as st
import os
from dotenv import load_dotenv
import git
import tempfile
import shutil
from pathlib import Path
import sys

# Load environment variables
load_dotenv()

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer
    from code_analyzer.rag_assistant import RAGCodeAssistant
except ImportError as e:
    st.error(f"Failed to import analyzer modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 LLM Code Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beast UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f093fb 0%, #f5576c 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .analysis-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .error-box {
        background: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>🤖 LLM Code Analyzer</h1><h3>AI-powered code beast with RAG</h3></div>', unsafe_allow_html=True)
st.title('🤖 LLM Code Analyzer')
st.subheader('AI-powered code beast with RAG')
try:
    st.markdown('''<style>
    html, body, [class^="st-"] {
        background-color: #000 !important;
        color: #00FF00 !important;
        font-family: "Courier New", Courier, monospace !important;
    }
    .stApp {
        background: url("https://www.transparenttextures.com/patterns/matrix.png") repeat !important;
        background-size: cover !important;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"],
    .stTextInput input,
    .stTextArea textarea {
        background-color: rgba(0,255,0,0.08) !important;
        color: #00FF00 !important;
        border: 1px solid #00FF00 !important;
        font-family: "Courier New", Courier, monospace !important;
    }
    .stButton > button {
        background-color: #00FF00 !important;
        color: #000 !important;
        border: 2px solid #00FF00 !important;
        font-family: "Courier New", Courier, monospace !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #009900 !important;
        color: #fff !important;
        border: 2px solid #00FF00 !important;
    }
    .stSidebar, .stSidebarContent, .css-1d391kg, .css-1lcbmhc {
        background-color: rgba(0,0,0,0.95) !important;
        color: #00FF00 !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #00FF00 !important;
        font-family: "Courier New", Courier, monospace !important;
        text-shadow: 0 0 8px #00FF00;
    }
    .stTabs [data-baseweb="tab"] {
        color: #00FF00 !important;
        background: #111 !important;
        border: 1px solid #00FF00 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #222 !important;
        color: #00FF00 !important;
        border-bottom: 2px solid #00FF00 !important;
    }
    .stAlert, .stInfo, .stSuccess, .stError, .stWarning {
        background-color: #111 !important;
        color: #00FF00 !important;
        border-left: 5px solid #00FF00 !important;
    }
    /* Digital rain animation */
    @keyframes fall {
        0% {transform: translateY(-100%); opacity: 0;}
        100% {transform: translateY(100%); opacity: 1;}
    }
    .matrix-rain {
        display: inline-block;
        animation: fall 2.5s linear infinite;
        color: #00FF00 !important;
        font-family: "Courier New", Courier, monospace !important;
        text-shadow: 0 0 8px #00FF00;
    }
    </style>''', unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Matrix theme CSS could not be applied: {e}")

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection with API key loading
    model = st.selectbox("Model", ["OpenAI", "Anthropic", "DeepSeek", "Mercury", "Gemini"])
    
    # Load API keys from environment
    api_keys = {
        "OpenAI": os.getenv('OPENAI_API_KEY'),
        "Anthropic": os.getenv('ANTHROPIC_API_KEY'),
        "DeepSeek": os.getenv('DEEPSEEK_API_KEY'),
        "Mercury": os.getenv('MERCURY_API_KEY'),
        "Gemini": os.getenv('GEMINI_API_KEY')
    }
    
    # Show API key status
    if api_keys[model]:
        st.success(f"✅ {model} API key found")
    else:
        st.error(f"❌ {model} API key missing")
    
    st.markdown("---")
    st.markdown("### 📝 Code Input")
    
    # Code input area in sidebar
    code_input = st.text_area(
        'Paste Code:',
        height=200,
        placeholder="Paste your code here...",
        help="Enter the code you want to analyze"
    )
    
    # File uploader in sidebar
    file_up = st.file_uploader(
        'Upload Code File:',
        type=['py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt'],
        help="Upload a code file to analyze"
    )
    
    # Handle file upload
    if file_up:
        try:
            code_input = file_up.read().decode('utf-8')
            st.success(f"✅ File '{file_up.name}' loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # GitHub repo URL input in sidebar
    repo_url = st.text_input(
        'GitHub Repo URL (optional):',
        placeholder="https://github.com/username/repo",
        help="Enter a GitHub repository URL to analyze the entire codebase"
    )
    
    if repo_url:
        st.info("🌐 Repository mode enabled")
    
    st.markdown("---")
    st.markdown("### 🚀 Actions")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button('🔍 Analyze', type="primary", use_container_width=True)
        fix_button = st.button('🔧 Fix Suggestions', use_container_width=True)
    with col2:
        security_button = st.button('🛡️ Security Check', use_container_width=True)
        performance_button = st.button('⚡ Performance Check', use_container_width=True)

# Main content area for analysis output
st.markdown('<div class="main-header"><h1>🤖 LLM Code Analyzer</h1><h3>AI-powered code analysis with RAG</h3></div>', unsafe_allow_html=True)

# Analysis logic
if analyze_button or fix_button or security_button or performance_button:
    if not code_input and not repo_url:
        st.error('❌ No code provided. Please paste code or upload a file.')
    else:
        try:
            with st.spinner('🤖 Initializing analyzer...'):
                analyzer = AdvancedCodeAnalyzer(model=model)
                rag = RAGCodeAssistant()
                # Run analysis based on button
                if analyze_button:
                    result = analyzer.analyze_code(code_input)
                    display_analysis_results(result, title="Analysis Results")
                elif fix_button:
                    # Fix suggestions logic (slimmed)
                    result = analyzer.analyze_code(code_input)
                    display_fix_suggestions(result)
                elif security_button:
                    result = analyzer.analyze_code(code_input)
                    display_security_results(result)
                elif performance_button:
                    result = analyzer.analyze_code(code_input)
                    st.markdown("### ⚡ Performance Report")
                    st.write(result)
        except Exception as e:
            st.error(f'❌ Analysis failed: {str(e)}')
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("**Error Details:**")
            st.code(str(e))
            st.markdown('</div>', unsafe_allow_html=True)

# Helper functions for displaying results
def display_analysis_results(result, title):
    """Display comprehensive analysis results."""
    st.markdown("---")
    st.markdown(f"### 📊 {title}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Summary", "🐛 Issues", "💡 Suggestions", "📈 Metrics"])
    
    with tab1:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Analysis Summary")
        # User-friendly summary rendering
        summary = None
        code_quality_score = None
        model_name = None
        execution_time = None
        # Try to extract from result or its attributes
        if hasattr(result, 'summary') and result.summary:
            summary = result.summary
        elif hasattr(result, 'code_analysis') and hasattr(result.code_analysis, 'documentation'):
            summary = result.code_analysis.documentation
        if hasattr(result, 'code_analysis') and hasattr(result.code_analysis, 'code_quality_score'):
            code_quality_score = result.code_analysis.code_quality_score
        if hasattr(result, 'code_analysis') and hasattr(result.code_analysis, 'model_name'):
            model_name = result.code_analysis.model_name
        if hasattr(result, 'code_analysis') and hasattr(result.code_analysis, 'execution_time'):
            execution_time = result.code_analysis.execution_time
        # Render nicely
        if summary:
            st.write(summary)
        else:
            st.info("No summary available for this code.")
        if code_quality_score is not None:
            st.metric("Code Quality Score", f"{code_quality_score}/100")
        if model_name:
            st.metric("Model Used", model_name)
        if execution_time is not None:
            st.metric("Analysis Time", f"{execution_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Potential Issues")
        if hasattr(result, 'potential_bugs'):
            for bug in result.potential_bugs:
                st.error(f"🐛 {bug}")
        else:
            st.info("No major issues detected!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Improvement Suggestions")
        if hasattr(result, 'improvement_suggestions'):
            for suggestion in result.improvement_suggestions:
                st.info(f"💡 {suggestion}")
        else:
            st.success("Code looks good! No major improvements needed.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Quality Metrics")
        if hasattr(result, 'code_analysis') and hasattr(result.code_analysis, 'code_quality_score'):
            st.metric("Quality Score", f"{result.code_analysis.code_quality_score}/100")
        if hasattr(result, 'analysis_duration'):
            st.metric("Analysis Time", f"{result.analysis_duration:.2f}s")
        if hasattr(result, 'features_used'):
            st.write("**Features Used:**")
            for feature in result.features_used:
                st.write(f"• {feature}")
        st.markdown('</div>', unsafe_allow_html=True)

def display_fix_suggestions(suggestions):
    """Display fix suggestions."""
    st.markdown("---")
    st.markdown("### 🔧 Fix Suggestions")
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"💡 Suggestion {i}: {suggestion.get('title', 'Code Improvement')}"):
                st.write(suggestion.get('explanation', ''))
                if suggestion.get('code'):
                    st.code(suggestion['code'], language='python')
                if suggestion.get('relevance_score'):
                    st.metric("Relevance", f"{suggestion['relevance_score']:.2f}")
    else:
        st.info("No specific fix suggestions found. Try running a comprehensive analysis.")

def display_security_results(security_result):
    """Display security analysis results."""
    st.markdown("---")
    st.markdown("### 🛡️ Security Analysis")
    
    if 'error' not in security_result:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.write(security_result['analysis'])
        st.metric("Model Used", security_result['model_used'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"Security analysis failed: {security_result['error']}")

def display_rag_results(search_results):
    """Display RAG search results."""
    st.markdown("---")
    st.markdown("### 📚 Codebase Context")
    
    if search_results:
        for i, result in enumerate(search_results, 1):
            with st.expander(f"📄 {result.snippet.file_path} (Score: {result.relevance_score:.2f})"):
                st.write(f"**Context:** {result.context}")
                st.write(f"**Explanation:** {result.explanation}")
                st.code(result.snippet.content, language='python')
    else:
        st.info("No similar code found in the codebase.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 Powered by Advanced AI Code Analysis | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)