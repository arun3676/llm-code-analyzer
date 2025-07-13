FROM python:3.11-slim
WORKDIR /app
ARG REQUIREMENTS_FILE=requirements.txt
COPY ${REQUIREMENTS_FILE} .
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE} --no-deps
COPY code_analyzer code_analyzer
EXPOSE 8080
CMD ["streamlit", "run", "code_analyzer/web/app.py", "--server.port", "8080", "--server.address", "0.0.0.0"] 