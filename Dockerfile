FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir kedro "kedro-datasets[pandas,pickle]" scikit-learn xgboost lightgbm pandas numpy fastapi uvicorn pyarrow

RUN mkdir -p data/02_intermediate \
    data/03_primary \
    data/05_model_input \
    data/06_models \
    data/07_model_output

RUN PYTHONPATH=src kedro run

ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "diabetes_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]