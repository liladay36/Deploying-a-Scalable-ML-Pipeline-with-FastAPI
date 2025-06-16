import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slice
)

# Fixture: Load a small sample dataset for testing
@pytest.fixture(scope="module")
def sample_data():
    # Load the full dataset
    df = pd.read_csv("data/census.csv")
    df_sample = df.sample(n=100, random_state=42)  # Use a small sample for speed
    return df_sample

# Common categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data_output_shape(sample_data):
    """
    Test that process_data correctly splits features and labels,
    and that the output shapes match expectations.
    """
    train, _ = train_test_split(sample_data, test_size=0.2, random_state=42)
    X, y, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert X.shape[0] == y.shape[0], "Features and labels should have the same number of rows"
    assert X.shape[1] > 0, "Processed features should have at least one column"


def test_model_training_and_inference(sample_data):
    """
    Test that the model can be trained and makes predictions of the correct shape.
    """
    train, test = train_test_split(sample_data, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert len(preds) == len(y_test), "Number of predictions should match number of test labels"
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"


def test_performance_on_slice(sample_data):
    """
    Test that the model can compute metrics on a data slice (e.g., sex = Male).
    """
    train, test = train_test_split(sample_data, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)

    slice_feature = "sex"
    slice_value = "Male"

    # ✅ Pass arguments positionally instead of by keyword
    p, r, f1 = performance_on_categorical_slice(
        test,
        slice_feature,
        slice_value,
        cat_features,  # <-- positional
        "salary",      # <-- label (also positional)
        encoder,
        lb,
        model,
    )

    assert 0.0 <= p <= 1.0, "Precision should be between 0 and 1"
    assert 0.0 <= r <= 1.0, "Recall should be between 0 and 1"
    assert 0.0 <= f1 <= 1.0, "F1-score should be between 0 and 1"
