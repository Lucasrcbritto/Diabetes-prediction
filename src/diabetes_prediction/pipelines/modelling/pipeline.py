from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_features, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_features,
            inputs="featured_data",
            outputs=["X_train", "X_test", "y_train", "y_test", "scaler"],
            name="prepare_features_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="trained_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test"],
            outputs=None,
            name="evaluate_model_node",
        ),
    ])