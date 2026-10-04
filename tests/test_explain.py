import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain import explain_prediction, EXPLAINABILITY_DISCLAIMER
from src.predict import predict_headline


def test_explain_prediction_valid_sports():
    """Verify explainability on authentic sports headline."""
    headline = "भारताने अंतिम सामन्यात विजय मिळवला"
    result = explain_prediction(headline)

    assert result["predicted_category"] == "Sports"
    assert result["model_probability"] > 0.4
    assert result["features_present_count"] > 0
    assert len(result["top_contributing_features"]) > 0

    # Ensure disclaimer is included
    assert result["disclaimer"] == EXPLAINABILITY_DISCLAIMER

    # Ensure predicted category matches predict_headline
    pred = predict_headline(headline)
    assert result["predicted_category"] == pred["predicted_label"]


def test_explain_prediction_valid_tech():
    """Verify explainability on code-mixed technology headline."""
    headline = "नवीन 5G स्मार्टफोन भारतात लाँच, किंमत फक्त"
    result = explain_prediction(headline)

    assert result["predicted_category"] == "Technology & Science"
    top_terms = [item["term"] for item in result["top_contributing_features"]]

    # Ensure key tech terms like '5g' or 'स्मार्टफोन' or 'लाँच' are present
    assert any(term in ["5g", "स्मार्टफोन", "लाँच", "किंमत"] for term in top_terms)


def test_explain_feature_contributions_are_numeric_and_sorted():
    """Verify contribution metrics are float and properly ordered."""
    headline = "मुख्यमंत्री एकनाथ शिंदे यांच्या उपस्थितीत बैठक"
    result = explain_prediction(headline)

    features = result["top_contributing_features"]
    assert len(features) > 0

    prev_contrib = float("inf")
    for feat in features:
        assert isinstance(feat["term"], str)
        assert isinstance(feat["contribution"], (float, int))
        assert isinstance(feat["tfidf"], (float, int))
        assert isinstance(feat["coefficient"], (float, int))
        assert feat["contribution"] <= prev_contrib
        prev_contrib = feat["contribution"]


def test_explain_features_belong_to_input_headline():
    """Verify that all extracted features are tokens/n-grams from the headline."""
    headline = "भारताने अंतिम सामन्यात विजय मिळवला"
    result = explain_prediction(headline)
    cleaned = result["preprocessed_text"].lower()

    for item in result["top_contributing_features"]:
        term = item["term"].lower()
        # For unigrams or bigrams, every word token must be part of the cleaned text
        for token in term.split():
            assert token in cleaned


def test_explain_empty_and_whitespace_input():
    """Verify that empty, None, and whitespace-only inputs raise ValueError."""
    with pytest.raises(ValueError):
        explain_prediction("")

    with pytest.raises(ValueError):
        explain_prediction("     ")

    with pytest.raises(ValueError):
        explain_prediction(None)

    with pytest.raises(ValueError):
        explain_prediction(12345)
