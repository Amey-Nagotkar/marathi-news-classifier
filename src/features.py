"""
TF-IDF Feature Engineering Module
Part of Marathi News Headline Classification System (Phase 4).

Provides reusable functions to create, fit, and transform text representations
using scikit-learn's TfidfVectorizer equipped with the custom Devanagari-safe
token pattern from src.preprocessing.

Critical Rules:
- The vectorizer must ALWAYS be fitted ONLY on training data (X_train).
- Test data (X_test) and user input must ONLY be transformed.
- lowercase=False because Latin lowercasing is already handled in Phase 2 preprocessing.
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import MARATHI_TOKEN_PATTERN


def create_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: bool = True,
    use_idf: bool = True,
    max_features: Optional[int] = None,
    lowercase: bool = False
) -> TfidfVectorizer:
    """
    Creates and configures a TfidfVectorizer specifically tailored for Marathi news text.

    Parameters
    ----------
    ngram_range : Tuple[int, int], default=(1, 2)
        The lower and upper boundary of the range of n-values for different n-grams.
    min_df : int or float, default=2
        Ignore terms that have a document frequency strictly lower than the given threshold.
    max_df : float or int, default=1.0
        Ignore terms that have a document frequency strictly higher than the given threshold.
    sublinear_tf : bool, default=True
        Apply sublinear tf scaling, replacing tf with 1 + log(tf).
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting.
    max_features : int, optional
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency.

    Returns
    -------
    TfidfVectorizer
        Configured, un-fitted TfidfVectorizer.
    """
    return TfidfVectorizer(
        token_pattern=MARATHI_TOKEN_PATTERN,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        use_idf=use_idf,
        max_features=max_features,
        lowercase=False  # Handled safely during Phase 2 preprocessing
    )


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    train_texts: pd.Series
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Fits the vectorizer strictly on training texts and returns the fitted vectorizer
    and the training feature matrix.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer instance.
    train_texts : pd.Series or iterable of str
        Cleaned training texts.

    Returns
    -------
    Tuple[TfidfVectorizer, csr_matrix]
        Fitted vectorizer and sparse training feature matrix.
    """
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train_tfidf


def transform_text(
    vectorizer: TfidfVectorizer,
    texts: Any
) -> csr_matrix:
    """
    Transforms arbitrary text(s) using an already-fitted vectorizer.
    Guarantees no modification to vocabulary or IDF weights.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer instance.
    texts : iterable of str or str
        Cleaned texts to transform.

    Returns
    -------
    csr_matrix
        Sparse feature matrix.
    """
    if isinstance(texts, str):
        texts = [texts]
    return vectorizer.transform(texts)


def transform_train_test(
    vectorizer: TfidfVectorizer,
    train_texts: pd.Series,
    test_texts: pd.Series
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Convenience function that enforces the strict data leakage rule:
    Fits on train_texts and transforms both train_texts and test_texts.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer instance.
    train_texts : pd.Series
        Cleaned training texts.
    test_texts : pd.Series
        Cleaned testing texts.

    Returns
    -------
    Tuple[csr_matrix, csr_matrix]
        (X_train_tfidf, X_test_tfidf)
    """
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf


def get_matrix_sparsity(matrix: csr_matrix) -> float:
    """
    Calculates the sparsity percentage of a sparse matrix.
    """
    total_elements = matrix.shape[0] * matrix.shape[1]
    if total_elements == 0:
        return 0.0
    non_zeros = matrix.nnz
    return 100.0 * (1.0 - (non_zeros / total_elements))
