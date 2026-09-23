FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-c", "import src.ingestion.loaders, src.ingestion.chunkers, src.generation.chain; print('ai-ml-systems ok')"]
