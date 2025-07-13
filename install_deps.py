import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

key_packages = [
    'langchain', 'chromadb', 'flask', 'openai', 'anthropic',
    'torch', 'dotenv', 'sentence_transformers', 'tiktoken', 'numpy',
    'requests', 'psutil'
]

missing = []
for pkg in key_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    logging.info(f'Installing missing packages: {" ".join(missing)}')
    # Uncomment the following line to actually install
    # subprocess.call([sys.executable, '-m', 'pip', 'install'] + missing)
else:
    logging.info('All key packages are installed.') 