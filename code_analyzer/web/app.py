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
    from code_analyzer.main import CodeAnalyzer
    from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer, AnalysisConfig
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
    
    # LLM Model selection
    llm_choice = st.selectbox(
        'LLM Model:',
        ['DeepSeek', 'Claude', 'OpenAI', 'Mercury'],
        help="Choose your preferred AI model for analysis"
    )
    
    # Analysis types
    analysis_types = st.multiselect(
        'Analysis Types:',
        ['Code Quality & Bugs', 'Performance Profiling', 'Security Scan', 'Framework-Specific', 'Cloud Integration', 'Container/K8s'],
        default=['Code Quality & Bugs'],
        help="Select the types of analysis you want to perform"
    )
    
    # Enable adaptive evaluations
    evals_on = st.checkbox(
        'Enable Adaptive Evals',
        value=True,
        help="Enable adaptive evaluations for more comprehensive analysis"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.info("Ready to analyze your code!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Code Input")
    
    # Code input area
    code_input = st.text_area(
        'Paste Code:',
        height=300,
        placeholder="Paste your code here or upload a file below...",
        help="Enter the code you want to analyze"
    )
    
    # File uploader
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

with col2:
    st.markdown("### 🔗 Repository")
    
    # GitHub repo URL input
    repo_url = st.text_input(
        'GitHub Repo URL (optional):',
        placeholder="https://github.com/username/repo",
        help="Enter a GitHub repository URL to analyze the entire codebase"
    )
    
    if repo_url:
        st.info("🌐 Repository mode enabled")

# Analysis button
st.markdown("---")
analyze_button = st.button(
    '🚀 Analyze Now',
    type="primary",
    use_container_width=True
)

# Analysis logic
if analyze_button:
    if not code_input and not repo_url:
        st.error('❌ No code, bro – input something.')
    else:
        try:
            with st.spinner('🤖 Initializing analyzer...'):
                # Initialize analyzer
                analyzer = CodeAnalyzer()
                # Map analysis types to config fields
                enable_performance = 'Performance Profiling' in analysis_types
                enable_security = 'Security Scan' in analysis_types
                enable_rag = 'Cloud Integration' in analysis_types or evals_on  # fallback to evals_on for RAG
                enable_multimodal = 'Container/K8s' in analysis_types
                config = AnalysisConfig(
                    enable_rag=enable_rag,
                    enable_performance=enable_performance,
                    enable_security=enable_security,
                    enable_multimodal=enable_multimodal
                )
                advanced = AdvancedCodeAnalyzer(config)
            
            # Handle repository analysis
            if repo_url:
                with st.spinner('📥 Cloning repository...'):
                    temp_dir = 'temp_repo'
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    
                    try:
                        git.Repo.clone_from(repo_url, temp_dir)
                        st.success(f"✅ Repository cloned successfully!")
                        
                        # Read all code files from the repository
                        code_input = ""
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                if file.endswith(('.py', '.js', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt')):
                                    file_path = os.path.join(root, file)
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            code_input += f"\n# File: {file}\n{f.read()}\n"
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not read {file}: {str(e)}")
                        
                        # Clean up temp directory
                        shutil.rmtree(temp_dir)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to clone repository: {str(e)}")
                        st.stop()
            
            # Perform analysis
            with st.spinner('🔍 Analyzing code...'):
                result = advanced.analyze_code_advanced(code_input, model=llm_choice.lower())
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Report")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Summary", "🐛 Issues", "💡 Suggestions", "📈 Metrics"])
            
            with tab1:
                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                st.markdown("#### Analysis Summary")
                if hasattr(result, 'summary'):
                    st.write(result.summary)
                else:
                    st.write(str(result))
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
                if hasattr(result, 'code_quality_score'):
                    st.metric("Quality Score", f"{result.code_quality_score}/100")
                if hasattr(result, 'execution_time'):
                    st.metric("Analysis Time", f"{result.execution_time:.2f}s")
                if hasattr(result, 'model_name'):
                    st.metric("Model Used", result.model_name)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Success message
            st.success("🎉 Analysis completed successfully!")
        
        except Exception as e:
            st.error(f'❌ Failed: {str(e)}')
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("**Error Details:**")
            st.code(str(e))
            st.markdown('</div>', unsafe_allow_html=True)

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