FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir blinker==1.8.2 -r requirements.txt && pip check
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"] 