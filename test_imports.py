import logging

logging.basicConfig(level=logging.INFO)

packages = [
    'langchain', 'chromadb', 'flask', 'openai', 'anthropic',
    'torch', 'dotenv', 'sentence_transformers', 'tiktoken', 'numpy',
    'requests', 'psutil'
]

for pkg in packages:
    try:
        __import__(pkg)
        logging.info(f'{pkg} imported successfully')
    except ImportError as e:
        logging.warning(f'Failed to import {pkg}: {e}') 