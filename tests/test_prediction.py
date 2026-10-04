"""
Unit tests for Phase 7 Production Prediction Engine (src/predict.py).
Verifies artifact loading, single headline prediction, probability properties,
input validation, batch predictions, and script robustness.
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_model, predict_headline, predict_batch
from src.train import LABELS_ORDER


def test_production_artifacts_exist():
    vec_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")
    clf_path = os.path.join(PROJECT_ROOT, "models", "final_classifier.joblib")
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

    assert os.path.exists(vec_path), f"Missing {vec_path}"
    assert os.path.exists(clf_path), f"Missing {clf_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"


def test_model_loading():
    vec, clf = load_model()
    assert vec is not None
    assert clf is not None
    assert len(clf.classes_) == 5
    assert set(clf.classes_) == set(LABELS_ORDER)


def test_canonical_prediction():
    text = "भारताने अंतिम सामन्यात विजय मिळवला"
    result = predict_headline(text)

    assert result['predicted_label'] == "Sports"
    assert 0.0 <= result['predicted_probability'] <= 1.0
    assert len(result['top_predictions']) == 3
    assert result['top_predictions'][0]['label'] == "Sports"

    # Probability sum check
    prob_sum = sum(result['all_class_probabilities'].values())
    assert abs(prob_sum - 1.0) < 0.01


def test_code_mixed_prediction():
    text = "नवीन 5G स्मार्टफोन भारतात लाँच, किंमत फक्त 10 हजार रुपये"
    result = predict_headline(text)

    assert result['predicted_label'] == "Technology & Science"
    assert result['predicted_probability'] > 0.50
    assert "Technology & Science" in [p['label'] for p in result['top_predictions']]


def test_top_predictions_sorting():
    text = "मुख्यमंत्री एकनाथ शिंदे यांच्या उपस्थितीत महत्त्वाची बैठक"
    result = predict_headline(text, top_k=5)

    probs = [p['probability'] for p in result['top_predictions']]
    # Verify strictly descending order
    assert probs == sorted(probs, reverse=True)


def test_input_validation_empty_and_whitespace():
    with pytest.raises(ValueError, match="Please enter a Marathi news headline"):
        predict_headline("")

    with pytest.raises(ValueError, match="Please enter a Marathi news headline"):
        predict_headline("   \t\n  ")

    with pytest.raises(ValueError, match="input was None"):
        predict_headline(None)

    with pytest.raises(ValueError, match="Headline must be text/string"):
        predict_headline(12345)


def test_predict_batch():
    texts = [
        "भारताने अंतिम सामन्यात विजय मिळवला",
        "नवीन 5G स्मार्टफोन भारतात लाँच",
        "   "  # Invalid entry to test graceful handling
    ]
    results = predict_batch(texts)
    assert len(results) == 3
    assert results[0]['predicted_label'] == "Sports"
    assert results[1]['predicted_label'] == "Technology & Science"
    assert "error" in results[2]
