FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git ffmpeg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "code_analyzer/web/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 