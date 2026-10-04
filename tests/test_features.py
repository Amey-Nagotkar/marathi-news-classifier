"""
Unit tests for Phase 4 TF-IDF Feature Engineering.
Verifies vectorizer creation, training fit, test transform, matrix shapes,
non-fragmentation of Devanagari words, and absence of data leakage.
"""

import os
import sys
import pytest
import pandas as pd
from scipy.sparse import csr_matrix

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    create_vectorizer,
    fit_vectorizer,
    transform_text,
    transform_train_test,
    get_matrix_sparsity
)
from src.preprocessing import MARATHI_TOKEN_PATTERN


@pytest.fixture
def sample_data():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')
    return df_train, df_test


def test_vectorizer_creation():
    vec = create_vectorizer(ngram_range=(1, 2), min_df=2, max_df=1.0)
    assert vec.ngram_range == (1, 2)
    assert vec.min_df == 2
    assert vec.max_df == 1.0
    assert vec.token_pattern == MARATHI_TOKEN_PATTERN
    assert vec.lowercase is False
    assert vec.sublinear_tf is True


def test_devanagari_words_not_fragmented():
    test_str = "भारताने अंतिम सामन्यात विजय मिळवला"
    vec = create_vectorizer(ngram_range=(1, 1), min_df=1)
    vec.fit([test_str])
    vocab = list(vec.vocabulary_.keys())

    assert "भारताने" in vocab
    assert "अंतिम" in vocab
    assert "सामन्यात" in vocab
    assert "विजय" in vocab
    assert "मिळवला" in vocab
    # Verify no broken sub-word fragments like 'रत' or 'मन'
    assert "रत" not in vocab
    assert "मन" not in vocab


def test_fit_on_train_only(sample_data):
    df_train, df_test = sample_data
    X_train = df_train['Cleaned_Text']
    X_test = df_test['Cleaned_Text']

    vec = create_vectorizer(ngram_range=(1, 2), min_df=2)
    vec, X_train_tfidf = fit_vectorizer(vec, X_train)
    X_test_tfidf = transform_text(vec, X_test)

    # Check matrix shapes
    assert X_train_tfidf.shape[0] == 11298
    assert X_test_tfidf.shape[0] == 2825
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1]
    assert X_train_tfidf.shape[1] == len(vec.vocabulary_)


def test_matrix_sparsity_calculation():
    mat = csr_matrix([[1, 0, 0], [0, 2, 0]])
    sparsity = get_matrix_sparsity(mat)
    # 2 non-zeros out of 6 elements -> 4/6 = 66.6667% sparse
    assert abs(sparsity - 66.6667) < 0.01


def test_code_mixed_and_number_features_preserved(sample_data):
    df_train, _ = sample_data
    X_train = df_train['Cleaned_Text']

    vec = create_vectorizer(ngram_range=(1, 2), min_df=2)
    vec, _ = fit_vectorizer(vec, X_train)
    vocab = vec.vocabulary_

    # Verify presence of Latin terms
    assert any('vs' in term for term in vocab), "Expected 'vs' in vocabulary"
    assert any('battery' in term for term in vocab), "Expected 'battery' in vocabulary"

    # Verify presence of numbers
    assert any('10' in term for term in vocab), "Expected '10' in vocabulary"


def test_no_empty_matrices(sample_data):
    df_train, df_test = sample_data
    vec = create_vectorizer(ngram_range=(1, 2), min_df=2)
    X_train_tfidf, X_test_tfidf = transform_train_test(
        vec, df_train['Cleaned_Text'], df_test['Cleaned_Text']
    )

    assert X_train_tfidf.nnz > 0
    assert X_test_tfidf.nnz > 0
