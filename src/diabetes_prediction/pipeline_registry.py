from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from diabetes_prediction.pipelines.data_engineering import create_pipeline as de
from diabetes_prediction.pipelines.modelling import create_pipeline as mod
from diabetes_prediction.pipelines.inference import create_pipeline as inf


def register_pipelines() -> dict[str, Pipeline]:
    data_engineering = de()
    modelling = mod()
    inference = inf()

    return {
        "data_engineering": data_engineering,
        "modelling": modelling,
        "inference": inference,
        "__default__": data_engineering + modelling + inference,
    }