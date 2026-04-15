from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_inference_data, predict
from diabetes_prediction.pipelines.data_engineering.nodes import clean_data, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs="raw_inference_data",
            outputs="cleaned_inference_data",
            name="clean_inference_node",
        ),
        node(
            func=feature_engineering,
            inputs="cleaned_inference_data",
            outputs="featured_inference_data",
            name="feature_engineering_inference_node",
        ),
        node(
            func=prepare_inference_data,
            inputs=["featured_inference_data", "X_train", "scaler"],
            outputs="prepared_inference_data",
            name="prepare_inference_node",
        ),
        node(
            func=predict,
            inputs=["trained_model", "prepared_inference_data"],
            outputs="inference_predictions",
            name="predict_node",
        ),
    ])