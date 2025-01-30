from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-code-analyzer",
    version="1.0.0",
    author="Arun Kumar Chukkala",
    author_email="arunkiran721@gmail.com",
    description="AI-powered code analysis system using multiple LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arun3676/llm-code-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)