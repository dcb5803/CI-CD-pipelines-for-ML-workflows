# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY main.py ./
RUN pip install pandas scikit-learn joblib

CMD ["python", "main.py"]
