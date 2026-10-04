"""
Unit and Integration Tests for Flask Web Application (Phase 8.6)
Tests web routing, REST API endpoints, prediction validation, and dataset statistics.
"""

import pytest
import json
from webapp.app import app, VALID_CATEGORIES


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page_status(client):
    """Verify that GET / returns HTTP 200 and loads classification interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Marathi News Classifier" in response.data
    assert b"Classify a Marathi News Headline" in response.data


def test_dataset_page_status(client):
    """Verify that GET /dataset returns HTTP 200 and loads dataset explorer."""
    response = client.get("/dataset")
    assert response.status_code == 200
    assert b"Explore the Marathi News Dataset" in response.data
    assert b"14,123" in response.data


def test_model_page_status(client):
    """Verify that GET /model returns HTTP 200 and displays benchmark KPIs."""
    response = client.get("/model")
    assert response.status_code == 200
    assert b"Model Performance &amp; Architecture" in response.data or b"Model Performance" in response.data
    assert b"89.84%" in response.data
    assert b"Balanced Logistic Regression" in response.data


def test_about_page_status(client):
    """Verify that GET /about returns HTTP 200 and contains project details."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About the Project" in response.data
    assert b"100% Local" in response.data


def test_api_predict_valid_marathi(client):
    """Test POST /api/predict with an authentic Marathi headline."""
    payload = {"headline": "भारताने अंतिम सामन्यात विजय मिळवला"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["success"] is True
    assert data["predicted_category"] in VALID_CATEGORIES
    assert data["predicted_category"] == "Sports"
    assert 0.0 <= data["model_probability"] <= 1.0
    assert len(data["top_predictions"]) == 3
    assert len(data["all_probabilities"]) == 5

    # Verify probability distribution sum ~ 1.0
    prob_sum = sum(data["all_probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01

    # Verify preprocessed text is returned
    assert "preprocessed_text" in data
    assert len(data["preprocessed_text"]) > 0

    # Verify explainability is included
    assert "explainability" in data
    assert "top_features" in data["explainability"]
    assert len(data["explainability"]["top_features"]) > 0
    assert "disclaimer" in data["explainability"]


def test_api_predict_code_mixed(client):
    """Test POST /api/predict with a code-mixed Marathi-English headline."""
    payload = {"headline": "नवीन 5G स्मार्टफोन भारतात लाँच, किंमत फक्त 15000"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["predicted_category"] == "Technology & Science"


def test_api_predict_empty_input(client):
    """Test POST /api/predict returns 400 for empty or whitespace strings."""
    # Empty string
    response = client.post(
        "/api/predict",
        data=json.dumps({"headline": ""}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["success"] is False
    assert "Please enter a Marathi news headline" in data["error"]

    # Whitespace only
    response_ws = client.post(
        "/api/predict",
        data=json.dumps({"headline": "   \n\t  "}),
        content_type="application/json"
    )
    assert response_ws.status_code == 400


def test_api_random_headline_all(client):
    """Test GET /api/random-headline with default category=all."""
    response = client.get("/api/random-headline")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "headline" in data
    assert len(data["headline"]) > 0
    assert data["category"] in VALID_CATEGORIES


def test_api_random_headline_specific_category(client):
    """Test GET /api/random-headline with category query parameter."""
    response = client.get("/api/random-headline?category=Sports")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["category"] == "Sports"


def test_api_random_headline_invalid_category(client):
    """Test GET /api/random-headline with an invalid category returns 400."""
    response = client.get("/api/random-headline?category=NonExistentCategory")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["success"] is False


def test_api_dataset_stats(client):
    """Test GET /api/dataset-stats returns accurate corpus statistics."""
    response = client.get("/api/dataset-stats")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["total_records"] == 14123
    assert data["total_categories"] == 5
    assert data["train_records"] == 11298
    assert data["test_records"] == 2825
    assert "category_counts" in data
    assert "category_percentages" in data
    assert len(data["category_counts"]) == 5


def test_api_category_explorer_valid(client):
    """Test GET /api/category/<category> returns category metrics and 5 real examples."""
    response = client.get("/api/category/Politics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["category"] == "Politics"
    assert data["record_count"] == 4029
    assert data["percentage"] == 28.53
    assert "avg_headline_length" in data
    assert data["avg_headline_length"] > 0
    assert len(data["examples"]) == 5


def test_api_category_explorer_invalid(client):
    """Test GET /api/category/<category> returns 404 for unknown category."""
    response = client.get("/api/category/Astronomy")
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data["success"] is False


def test_api_explain_endpoint(client):
    """Test POST /api/explain returns feature contributions."""
    response = client.post(
        "/api/explain",
        data=json.dumps({"headline": "भारताने अंतिम सामन्यात विजय मिळवला"}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert data["predicted_category"] == "Sports"
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) > 0
    assert "disclaimer" in data
