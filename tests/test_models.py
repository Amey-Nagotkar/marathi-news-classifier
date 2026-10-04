"""
Unit tests for Phase 5 Model Training and Comparison.
Verifies model training, predictions, metrics, confusion matrix dimensions,
and probability support.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import (
    load_data,
    get_candidate_models,
    evaluate_model,
    LABELS_ORDER
)
from src.features import create_vectorizer, fit_vectorizer, transform_text


@pytest.fixture(scope="module")
def prepared_data():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    vectorizer = create_vectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        sublinear_tf=True,
        use_idf=True,
        lowercase=False
    )
    vectorizer, X_train_tfidf = fit_vectorizer(vectorizer, X_train)
    X_test_tfidf = transform_text(vectorizer, X_test)

    return X_train_tfidf, y_train, X_test_tfidf, y_test


def test_model_results_csv_exists():
    csv_path = os.path.join(PROJECT_ROOT, "reports", "model_results.csv")
    if not os.path.exists(csv_path):
        pytest.skip("reports/ removed during cleanup for deployment")
    df_res = pd.read_csv(csv_path)
    assert len(df_res) == 5, f"Expected 5 models in results CSV, found {len(df_res)}"
    assert "Accuracy" in df_res.columns
    assert "Macro_F1" in df_res.columns


def test_all_models_train_and_predict(prepared_data):
    X_train_tfidf, y_train, X_test_tfidf, y_test = prepared_data
    models = get_candidate_models()
    assert len(models) == 5

    for name, model in models.items():
        res = evaluate_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
        
        # Predictions check
        assert len(res['y_pred']) == len(y_test)
        assert set(res['y_pred']).issubset(set(LABELS_ORDER))
        
        # Metric bounds
        assert 0.0 <= res['accuracy'] <= 1.0
        assert 0.0 <= res['macro_f1'] <= 1.0
        assert 0.0 <= res['weighted_f1'] <= 1.0
        
        # Confusion matrix check
        cm = res['confusion_matrix']
        assert cm.shape == (5, 5)
        assert cm.sum() == len(y_test)


def test_probability_support(prepared_data):
    X_train_tfidf, y_train, X_test_tfidf, y_test = prepared_data
    models = get_candidate_models()

    # Logistic Regression
    lr = models["Logistic Regression"]
    lr.fit(X_train_tfidf, y_train)
    probs = lr.predict_proba(X_test_tfidf[:10])
    assert probs.shape == (10, 5)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), rtol=1e-5)

    # Linear SVM should NOT have predict_proba
    svm = models["Linear SVM"]
    svm.fit(X_train_tfidf, y_train)
    assert not hasattr(svm, "predict_proba")


def test_confusion_matrix_figures_exist():
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    if not os.path.exists(fig_dir):
        pytest.skip("reports/ removed during cleanup for deployment")
    expected_figures = [
        "multinomial_naive_bayes_cm.png",
        "logistic_regression_cm.png",
        "logistic_regression_balanced_cm.png",
        "linear_svm_cm.png",
        "linear_svm_balanced_cm.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(fig_dir, fig_name)
        assert os.path.exists(fig_path), f"Missing confusion matrix plot: {fig_path}"
        assert os.path.getsize(fig_path) > 1000, f"Plot file seems empty: {fig_path}"
