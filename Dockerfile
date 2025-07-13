FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements_force.txt .
RUN pip install --no-cache-dir -r requirements_force.txt --no-deps
COPY code_analyzer/web/app.py code_analyzer/web/
COPY code_analyzer/main.py code_analyzer/
COPY code_analyzer/advanced_analyzer.py code_analyzer/
EXPOSE 8080
CMD ["streamlit", "run", "code_analyzer/web/app.py", "--server.port", "8080", "--server.address", "0.0.0.0"] 