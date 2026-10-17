
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY deployment/ ./deployment/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
