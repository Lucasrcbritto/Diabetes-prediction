from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs="raw_diabetes_data",
            outputs="cleaned_data",
            name="clean_data_node",
        ),
        node(
            func=feature_engineering,
            inputs="cleaned_data",
            outputs="featured_data",
            name="feature_engineering_node",
        ),
    ])