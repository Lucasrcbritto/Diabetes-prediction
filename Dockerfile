FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir kedro scikit-learn xgboost lightgbm pandas numpy fastapi uvicorn pyarrow

ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "diabetes_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]