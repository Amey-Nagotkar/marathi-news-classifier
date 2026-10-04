# Marathi News Classifier

An NLP-based machine learning system for classifying Marathi news headlines into five topical categories using sublinear TF-IDF feature representations and Balanced Logistic Regression.

This repository represents an academic NLP/ML project developed for Marathi text processing, featuring custom Devanagari tokenization, class-imbalance mitigation, linear model explainability, and a modern, 100% local Flask web application.

---

## Overview

Regional Indian language news headlines present unique natural language processing challenges: high code-mixing with English terminology, extensive morphological inflections, script-specific combining diacritics (matras), and significant topical class imbalance. Manually categorizing news headlines at scale is time-consuming and inefficient.

This system automates topical categorization by classifying Marathi news headlines into five discrete classes:

1. **Technology & Science** (तंत्रज्ञान आणि विज्ञान)
2. **Politics** (राजकारण)
3. **Business** (अर्थकारण / व्यवसाय)
4. **Sports** (क्रीडा)
5. **Entertainment** (मनोरंजन)

---

## Key Features

- **Automated Marathi Headline Classification:** Fast, deterministic topical categorization of arbitrary Marathi and code-mixed headlines.
- **Custom Devanagari Tokenization:** Regex-based tokenizer that avoids word fragmentation at combining vowel signs (matras).
- **Sublinear TF-IDF Feature Representation:** 29,455 unigram and bigram features capturing atomic words and domain-specific phrases.
- **Balanced Multi-Class Logistic Regression:** Inversely weighted class balancing to handle severe corpus skew (e.g., Business at 4.93% vs. Technology at 44.62%).
- **Calibrated Model Probabilities:** Full probability distribution across all 5 classes via `predict_proba()`.
- **Top-3 Ranked Predictions:** Hierarchical probability output displaying dominant and runner-up classifications.
- **Feature-Level Model Explainability:** Mathematical decomposition ($x_i \cdot w_{c,i}$) highlighting observed terms with the strongest positive contributions toward the predicted class.
- **Interactive Classification Pipeline:** Visual 5-stage pipeline diagram with an on-demand parameter inspector.
- **Dataset Explorer:** Dedicated interface showing corpus profile metrics, class distribution charts, and authentic sample headlines for each category.
- **Random Dataset Headline Generator:** Instant sampling of verified headlines across all categories directly into the classifier.
- **Model Comparison & Confusion Matrix:** Detailed benchmark comparison of 5 candidate architectures with normalized confusion matrix analysis.
- **Prediction Session History & Export:** Client-side tracking of session classifications with one-click CSV export.
- **Responsive Web Interface:** Custom-styled Flask frontend with deep navy surfaces, SVG donut charts, and fluid micro-interactions.
- **100% Offline Local Execution:** Operates entirely on local hardware with zero external AI APIs, cloud subscriptions, or tracking dependencies.

---

## NLP Pipeline

```text
Raw Marathi Headline
        ↓
Text Preprocessing (Devanagari Normalization, NFC, Cleanup)
        ↓
TF-IDF Vectorization (29,455 Unigram + Bigram Features)
        ↓
Balanced Logistic Regression (One-vs-Rest / Softmax Estimation)
        ↓
Category Prediction + Calibrated Model Probability
        ↓
Feature Explainability (Term Contribution Decomposition)
```

1. **Raw Ingestion:** Ingests raw news headline text from user input or dataset samples.
2. **Preprocessing:** Cleans typography, normalizes Unicode, and standardizes spacing while preserving Marathi script integrity.
3. **Feature Extraction:** Projects the cleaned string into a 29,455-dimensional TF-IDF feature space using custom tokenization.
4. **Classification:** Evaluates the linear decision function using balanced class weights.
5. **Probability & Ranking:** Generates calibrated probabilities across all five classes and identifies the top-3 predictions.
6. **Explainability:** Calculates the term-level contribution scores of present vocabulary features for transparent model interpretation.

---

## Preprocessing

The preprocessing module ([src/preprocessing.py](src/preprocessing.py)) implements an authentic, non-destructive normalization pipeline tailored to Devanagari text:

- **HTML Entity Unescaping:** Decodes entities like `&amp;`, `&quot;`, `&#39;` using Python's `html.unescape`.
- **Unicode Canonical Composition (NFC):** Normalizes combining characters and vowel signs via `unicodedata.normalize('NFC', text)` to eliminate representation discrepancies.
- **Zero-Width & Control Character Removal:** Strips byte-order marks (`\ufeff`) and zero-width spaces/joiners (`\u200b`, `\u200e`, `\u200f`).
- **Whitespace & Non-Breaking Space Normalization:** Standardizes non-breaking spaces (`\u00a0`) and collapses redundant tabs and spaces.
- **Typographical Standardization:** Maps curly quotes (`“”‘’«»`) and em/en dashes (`–—―−`) to standard ASCII characters (`"`, `'`, `-`).
- **Script Preservation:** Explicitly preserves Marathi Devanagari letters, matras, conjuncts, English/Latin loanwords, ASCII digits (`0-9`), and Devanagari digits (`०-९`).
- **Latin Lowercasing:** Converts Latin characters to lowercase while leaving case-less Devanagari characters untouched.
- **No Stop-Word Removal:** Empirical experimentation confirmed that removing Marathi stop-words strips critical contextual markers in concise news headlines, reducing classification accuracy. All tokens are retained.

---

## Feature Engineering

Text representations are generated using scikit-learn's `TfidfVectorizer` configured specifically for Indic text:

- **Token Pattern:** `r'[\u0900-\u097F\w]+'`  
  Standard regex patterns (`\b\w\w+\b`) incorrectly split Devanagari words at vowel signs. This custom pattern keeps complex Marathi conjuncts intact alongside English loanwords and numbers.
- **N-gram Range:** Unigrams + Bigrams `(1, 2)`  
  Captures both single keywords (e.g., `सामना`, `निवडणूक`) and essential compound noun phrases (e.g., `अंतिम सामना`, `विधानसभा निवडणूक`).
- **Sublinear Term Frequency:** `sublinear_tf=True`  
  Replaces raw term frequency $tf$ with $1 + \log(tf)$ to dampen the impact of repeated words in longer headlines.
- **Document Frequency Cutoffs:** `min_df=2`, `max_df=1.0`  
  Discards rare single-occurrence typos while retaining corpus-wide domain vocabulary.
- **Production Vocabulary Size:** **29,455** total features extracted from the verified training corpus.

---

## Dataset

The corpus consists of **14,123** authentic Marathi news headlines consolidated from regional news publications. The final five-class dataset was curated by consolidating overlapping domains and excluding out-of-scope categories (Crime and Fashion) to ensure coherent, distinct target labels.

- **Total Verified Records:** 14,123
- **Number of Categories:** 5
- **Stratified Training Split (80%):** 11,298 records
- **Stratified Held-Out Testing Split (20%):** 2,825 records
- **Data Integrity:** 0 missing values, 0 duplicate records, 0 train/test data leakage.

### Dataset Distribution

| Category | Total Records | Corpus Share | Training Split (80%) | Testing Split (20%) |
|:---|---:|---:|---:|---:|
| **Technology & Science** | 6,301 | 44.62% | 5,041 | 1,260 |
| **Politics** | 4,029 | 28.53% | 3,223 | 806 |
| **Sports** | 1,679 | 11.89% | 1,343 | 336 |
| **Entertainment** | 1,418 | 10.04% | 1,134 | 284 |
| **Business** | 696 | 4.93% | 557 | 139 |
| **Total Corpus** | **14,123** | **100.00%** | **11,298** | **2,825** |

---

## Model Evaluation

The production model was benchmarked on the untouched **2,825** held-out test records using stratified 5-fold validation principles:

- **Test Accuracy:** **89.84%**
- **Macro F1-Score:** **86.75%**
- **Weighted F1-Score:** **89.84%**

### Per-Class Test Performance (Balanced Logistic Regression)

| Category | Precision | Recall | F1-Score | Support |
|:---|---:|---:|---:|---:|
| **Technology & Science** | 92.42% | 90.71% | 91.57% | 1,260 |
| **Politics** | 88.58% | 91.19% | 89.85% | 806 |
| **Business** | 72.80% | 71.22% | 72.00% | 139 |
| **Sports** | 93.37% | 90.77% | 92.10% | 336 |
| **Entertainment** | 86.85% | 89.44% | 88.21% | 284 |

*Note: Class balancing (`class_weight='balanced'`) significantly boosted minority class recall for Business from 51.08% up to 71.22%.*

---

## Model Comparison

Five distinct model architectures were trained and objectively compared on the identical training and testing splits:

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Supports `predict_proba()` | Status |
|:---|---:|---:|---:|:---:|:---:|
| **Multinomial Naive Bayes** | 85.66% | 72.80% | 84.70% | Yes | Baseline |
| **Logistic Regression (Default)** | 88.64% | 82.34% | 88.24% | Yes | Evaluated |
| **Balanced Logistic Regression** | **89.84%** | **86.75%** | **89.84%** | **Yes** | **Selected (Production)** |
| **Linear SVM (Default)** | 91.33% | 88.71% | 91.30% | No (Hinge Loss) | Evaluated |
| **Balanced Linear SVM** | 91.68% | 88.96% | 91.70% | No (Hinge Loss) | Evaluated |

### Why Balanced Logistic Regression?

Balanced Linear SVM achieved the highest raw test accuracy at **91.68%** (compared to 89.84% for Balanced Logistic Regression). 

However, **Balanced Logistic Regression** was selected as the final production deployment model because:
1. It natively exposes calibrated probabilistic estimates via `predict_proba()` through the softmax/logistic function, which is essential for the application's multi-class probability visualizer and top-3 prediction ranking.
2. Linear SVM with hinge loss produces non-probabilistic margin distances that do not naturally map to calibrated probabilities without costly external sigmoid fitting (Platt scaling).
3. Logistic Regression delivers a clean balance of high accuracy (89.84%), strong minority class recovery (71.22% Business recall), and straightforward mathematical explainability.

*Model probabilities represent the model's relative classification scores across the five target classes and should not be construed as absolute certainty.*

---

## Explainability

To move beyond black-box predictions, the system includes a direct mathematical feature contribution decomposition ([src/explain.py](src/explain.py)).

For a given input vector $\mathbf{x}$ and predicted category $c$, the contribution score of each observed token $t_i$ is computed as:

$$\text{Contribution}(t_i) = x_i \cdot w_{c, i}$$

where:
- $x_i$ is the TF-IDF weight of token $t_i$ in the headline.
- $w_{c, i}$ is the learned Logistic Regression weight vector coefficient for category $c$ and feature $i$.

The web application visualizes the top positive contributing terms alongside their respective contribution magnitudes. These scores reflect the internal statistical weights learned by the linear model and are intended for technical interpretability rather than cognitive human reasoning.

---

## Web Application

The user-facing interface is built with **Flask**, **HTML5**, **Vanilla CSS**, and **JavaScript** (zero heavy frontend frameworks, zero CDN dependencies):

- **Classify Workspace:** Headline input area with real-time linguistic diagnostics (word count, character count, Devanagari %, Latin %, digits).
- **Interactive Prediction Reveal:** Instant classification animation showing dominant category, pure SVG radial donut probability chart, and high-contrast class bars.
- **Explainability Panel:** Term contribution chart explaining the primary linguistic features influencing the prediction.
- **Interactive NLP Pipeline:** Clickable 5-stage architectural diagram detailing tokenization, vectorization, and model parameters.
- **Dataset Explorer:** Category overview with interactive tabs revealing authentic corpus headlines.
- **Model Dashboard:** Comprehensive benchmark tables, selection justification, and confusion matrix explanation.
- **Session History & CSV Export:** Client-side tracking of classified headlines with one-click export.

---

## Project Structure

```text
Marathi News Classifier/
├── Master_Final_Marathi_News_5Label.csv    # Master consolidated 5-class dataset (14,123 rows)
├── requirements.txt                        # Minimal Python runtime dependencies
├── .gitignore                              # Git exclusion rules for caches, envs, and temporary files
├── README.md                               # Project documentation
│
├── data/
│   ├── train.csv                           # 80% stratified training split (11,298 rows)
│   └── test.csv                            # 20% stratified held-out testing split (2,825 rows)
│
├── models/
│   ├── final_classifier.joblib             # Serialized Balanced Logistic Regression model
│   ├── tfidf_vectorizer.joblib             # Serialized fitted TfidfVectorizer (29,455 features)
│   ├── marathi_news_classifier.joblib      # Combined pipeline artifact
│   └── model_metadata.json                 # Model hyperparameters, vocabulary size & metrics
│
├── src/
│   ├── preprocessing.py                    # Devanagari text normalization and tokenization
│   ├── features.py                         # TF-IDF vectorizer configuration and feature extraction
│   ├── train.py                            # Candidate model training and evaluation routines
│   ├── train_production.py                 # Production model training on full corpus
│   ├── evaluate.py                         # Stratified test benchmark evaluation routines
│   ├── predict.py                          # Production inference module (single & batch)
│   ├── explain.py                          # Linear feature contribution decomposition
│   └── split_data.py                       # Stratified train/test splitting logic
│
├── tests/
│   ├── test_data_split.py                  # Tests for dataset row counts, split, and stratification
│   ├── test_features.py                    # Tests for vectorizer, tokenization, and sparsity
│   ├── test_models.py                      # Tests for model training and probability support
│   ├── test_prediction.py                  # Tests for single, batch, and code-mixed inference
│   ├── test_explain.py                     # Tests for feature explainability calculation
│   ├── test_flask_app.py                   # Tests for web routes, views, and REST API endpoints
│   └── test_evaluation.py                  # Tests for metric calculation routines
│
└── webapp/
    ├── app.py                              # Flask application server and REST API handlers
    ├── static/
    │   ├── css/style.css                   # Custom design system, typography, and responsive styles
    │   ├── js/app.js                       # Frontend logic, SVG donut chart, and API interactions
    │   └── assets/confusion_matrix.png     # Evaluation confusion matrix visualization
    └── templates/
        ├── base.html                       # Base layout with sticky navbar and ambient canvas
        ├── index.html                      # Main classification workspace and prediction reveal
        ├── dataset.html                    # Dataset explorer and category tabs
        ├── model.html                      # Model performance, benchmark table, and architecture
        └── about.html                      # Academic problem overview and methodology
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** (Developed and validated on Python 3.12)
- Standard web browser (Chrome, Edge, Firefox, Safari)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/marathi-news-classifier.git
   cd "marathi-news-classifier"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application

Start the local Flask development server:
```bash
python webapp/app.py
```

Once running, navigate to the local address in your web browser:
```text
http://127.0.0.1:5000
```

### Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## Technical Summary

| Component | Technical Specification |
|:---|:---|
| **Domain** | Natural Language Processing (NLP) / Text Classification |
| **Language** | Marathi (मराठी) + Marathi-English Code-Mixed |
| **Corpus Size** | 14,123 News Headlines (100% verified authentic) |
| **Vocabulary** | 29,455 Unigrams and Bigrams |
| **Feature Extraction** | Sublinear TF-IDF (`sublinear_tf=True`, `min_df=2`) |
| **Classifier** | Logistic Regression (`class_weight='balanced'`, `max_iter=2000`) |
| **Test Accuracy** | **89.84%** (Macro F1: **86.75%**) |
| **Serving Layer** | Flask (Python 3.12) with REST API |
| **Deployment Mode** | 100% Offline Local Execution (No External AI APIs) |
