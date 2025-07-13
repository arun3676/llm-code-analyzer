import pkg_resources
import logging

logging.basicConfig(level=logging.INFO)

# Log all installed packages
packages = {p.project_name: p.version for p in pkg_resources.working_set}
logging.info(f'Deps: {packages}')

# Slim set of required dependencies
slim_deps = [
    'streamlit==1.36.0',
    'langchain==0.2.7',
    'chromadb==0.5.3',
    'openai==1.35.10',
    'anthropic==0.30.0',
    'python-dotenv==1.0.1',
    'gitpython==3.1.43',
    'peft==0.11.1',
    'bitsandbytes==0.43.1',
    'torch==2.3.1+cpu',
    'transformers==4.42.3',
    'pillow==10.4.0',
]
logging.info(f'Slim Deps: {slim_deps}') 