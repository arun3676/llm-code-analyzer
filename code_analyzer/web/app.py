import streamlit as st
import os
from dotenv import load_dotenv
import git
import tempfile
import shutil
from pathlib import Path
import sys
import json

# Load environment variables
load_dotenv()

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the required modules
try:
    from code_analyzer.advanced_analyzer import AdvancedCodeAnalyzer
except ImportError as e:
    st.error(f"Failed to import analyzer modules: {e}")
    st.stop()

# Helper functions for displaying results

def display_analysis_results(result, title):
    """Display comprehensive analysis results."""
    st.markdown("---")
    st.markdown(f"### 📊 {title}")
    
    summary = getattr(result, 'documentation', "No summary available.")
    code_quality_score = getattr(result, 'code_quality_score', None)
    model_name = getattr(result, 'model_name', 'N/A')
    execution_time = getattr(result, 'execution_time', 0)
    potential_bugs = getattr(result, 'potential_bugs', [])
    improvement_suggestions = getattr(result, 'improvement_suggestions', [])

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Summary", "🐛 Issues", "💡 Suggestions", "📈 Metrics"])
    
    with tab1:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Analysis Summary")
        st.write(summary)
        if code_quality_score is not None:
            st.metric("Code Quality Score", f"{code_quality_score}/100")
        st.metric("Model Used", model_name)
        st.metric("Analysis Time", f"{execution_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Potential Issues")
        if potential_bugs:
            for bug in potential_bugs:
                st.error(f"🐛 {bug}")
        else:
            st.info("No major issues detected!")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with tab3:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Improvement Suggestions")
        if improvement_suggestions:
            for suggestion in improvement_suggestions:
                st.info(f"💡 {suggestion}")
        else:
            st.success("Code looks good! No major improvements needed.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
        st.markdown("#### Quality Metrics")
        if code_quality_score is not None:
            st.metric("Quality Score", f"{code_quality_score}/100")
        st.metric("Analysis Time", f"{execution_time:.2f}s")
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

def display_multimodal_results(result, model_name):
    """Displays the results from the multimodal analysis."""
    st.markdown(f"#### Results from {model_name}")
    if "error" in result:
        st.error(f"Analysis failed: {result['error']}")
        return
    
    st.write(result.get('analysis', 'No analysis text available.'))
    
    if result.get('code_extracted'):
        st.code(result['code_extracted'], language='python')
    
    if result.get('suggestions'):
        st.markdown("##### Suggestions")
        for suggestion in result['suggestions']:
            st.info(f"💡 {suggestion}")

# Page configuration
st.set_page_config(
    page_title="🤖 LLM Code Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the Matrix theme
st.markdown("""
<canvas id="matrix-canvas"></canvas>
<style>
    /* General Body Styles */
    body, .stApp {
        background-color: #000 !important;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }

    #matrix-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }

    /* Main Header */
    .main-header {
        background: rgba(0, 0, 0, 0.5);
        border: 2px solid #00ff41;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 20px #00ff41;
        backdrop-filter: blur(5px);
    }
    .main-header h1, .main-header h3 {
        color: #00ff41;
        text-shadow: 0 0 7px #00ff41;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: rgba(10, 10, 10, 0.8);
        border-right: 1px solid #00ff41;
        backdrop-filter: blur(5px);
    }

    /* Code Input Area */
    .stTextArea, .stTextInput {
        position: relative;
        background: transparent;
    }
    .stTextArea > div > textarea, .stTextInput > div > div > input {
        background: rgba(10, 20, 10, 0.85);
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 5px;
        box-shadow: inset 0 0 10px #00ff41;
        font-size: 1.1rem !important;
        line-height: 1.6;
    }

    /* Cyber Buttons */
    .stButton > button {
        background: transparent;
        border: 2px solid #00ff41;
        color: #00ff41;
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Courier New', monospace;
        position: relative;
        cursor: pointer;
        transition: all 0.3s;
        overflow: hidden;
    }
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, transparent, rgba(0, 255, 65, 0.3), transparent);
        transition: all 0.5s;
    }
    .stButton > button:hover:before {
        left: 100%;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px #00ff41;
        color: #fff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 0, 0, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for the Matrix effect
st.components.v1.html("""
<script>
    const canvas = document.getElementById('matrix-canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%+-/~{[|`]}';
    const fontSize = 16;
    const columns = canvas.width / fontSize;

    const drops = [];
    for(let x = 0; x < columns; x++) {
        drops[x] = 1;
    }

    function draw() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#00ff41';
        ctx.font = fontSize + 'px monospace';

        for(let i = 0; i < drops.length; i++) {
            const text = letters.charAt(Math.floor(Math.random() * letters.length));
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);

            if(drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }

    setInterval(draw, 33);
</script>
""", height=0)

# Main header (keep only one - no duplicate titles)
st.markdown('<div class="main-header"><h1>🤖 LLM Code Analyzer</h1><h3>AI-powered code beast</h3></div>', unsafe_allow_html=True)

# Main content area with tabs
tab1, tab2 = st.tabs(["Code Analysis", "Multimodal Analysis"])

with tab1:
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        # Model selection only
        model = st.selectbox("Model", ["OpenAI", "Anthropic", "DeepSeek", "Mercury", "Gemini"])

    # Centered code input and actions
    st.markdown("""
        <style>
        .centered-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 2rem;
        }
        .big-textarea textarea {
            min-height: 300px !important;
            font-size: 1.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="centered-box">', unsafe_allow_html=True)
        code_input = st.text_area(
            'Paste your code here:',
            height=300,
            placeholder="Paste your code here...",
            help="Enter the code you want to analyze",
            key="main_code_input"
        )
        st.markdown("---")
        analyze_button = st.button('🔍 Analyze', type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # API key status (show below model selector if desired, or in main area)
    api_keys = {
        "OpenAI": os.getenv('OPENAI_API_KEY'),
        "Anthropic": os.getenv('ANTHROPIC_API_KEY'),
        "DeepSeek": os.getenv('DEEPSEEK_API_KEY'),
        "Mercury": os.getenv('MERCURY_API_KEY'),
        "Gemini": os.getenv('GEMINI_API_KEY')
    }
    if not api_keys[model]:
        st.error(f"❌ {model} API key missing")
    else:
        st.success(f"✅ {model} API key found")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'code_input' not in st.session_state:
        st.session_state.code_input = ""

    # Analysis logic
    if analyze_button:
        if not code_input:
            st.error('❌ No code provided. Please paste code or upload a file.')
        else:
            try:
                with st.spinner('🤖 Analyzing your code... Stand by...'):
                    analyzer = AdvancedCodeAnalyzer(model=model)
                    result = analyzer.analyze_code_advanced(code_input, model=model)
                    st.session_state.analysis_result = result
                    st.session_state.code_input = code_input
                    st.session_state.chat_history = [] # Reset chat on new analysis
            except Exception as e:
                st.error(f'❌ Analysis failed: {str(e)}')
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.markdown("**Error Details:**")
                st.code(str(e))
                st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.analysis_result:
        display_analysis_results(st.session_state.analysis_result, title="Analysis Results")
        
        st.markdown("---")
        st.markdown("### 💬 Chat With Your Code")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your code..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    analyzer = AdvancedCodeAnalyzer(model=model)
                    response = analyzer.chat_with_code(
                        st.session_state.code_input, 
                        st.session_state.chat_history, 
                        prompt
                    )
                    st.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()


with tab2:
    st.header("🖼️ Multimodal Analysis")
    uploaded_file = st.file_uploader("Upload an image (screenshot, diagram, etc.)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        
        prompt = st.text_input("Analysis prompt (optional)", "Analyze this image and extract any relevant information or code.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            analyze_gemini = st.button("Analyze with Gemini Vision")
        with col2:
            analyze_claude = st.button("Analyze with Claude")
        with col3:
            analyze_all = st.button("Analyze with All Models")

        if analyze_gemini:
            with st.spinner("Analyzing with Gemini Vision..."):
                analyzer = AdvancedCodeAnalyzer()
                result = analyzer.analyze_image(uploaded_file, prompt, model='gemini-vision')
                display_multimodal_results(result, "Gemini Vision")

        if analyze_claude:
            with st.spinner("Analyzing with Claude..."):
                analyzer = AdvancedCodeAnalyzer()
                result = analyzer.analyze_image(uploaded_file, prompt, model='claude')
                display_multimodal_results(result, "Claude")

        if analyze_all:
            with st.spinner("Analyzing with all available models..."):
                analyzer = AdvancedCodeAnalyzer()
                results = analyzer.analyze_image_all(uploaded_file, prompt)
                for model_name, result in results.items():
                    if 'error' not in result:
                        display_multimodal_results(result, model_name)
                    else:
                        st.error(f"Failed to get analysis from {model_name}: {result['error']}")

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