"""
Unit tests for Phase 3 Data Splitting.
Verifies split sizes, class stratification, leakage prevention, and file integrity.
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_split_files_exist():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    assert os.path.exists(train_path), "train.csv does not exist"
    assert os.path.exists(test_path), "test.csv does not exist"


def test_split_row_counts():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    assert len(df_train) == 11298, f"Expected 11298 train rows, got {len(df_train)}"
    assert len(df_test) == 2825, f"Expected 2825 test rows, got {len(df_test)}"
    assert len(df_train) + len(df_test) == 14123, "Total rows do not equal 14123"


def test_no_missing_values():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    assert df_train['Text'].isnull().sum() == 0, "Nulls found in train Text"
    assert df_train['Label'].isnull().sum() == 0, "Nulls found in train Label"
    assert df_test['Text'].isnull().sum() == 0, "Nulls found in test Text"
    assert df_test['Label'].isnull().sum() == 0, "Nulls found in test Label"


def test_all_five_classes_present():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    expected_classes = {
        'Technology & Science',
        'Politics',
        'Business',
        'Sports',
        'Entertainment'
    }
    assert set(df_train['Label'].unique()) == expected_classes
    assert set(df_test['Label'].unique()) == expected_classes


def test_no_data_leakage_between_splits():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    train_texts = set(df_train['Cleaned_Text'])
    test_texts = set(df_test['Cleaned_Text'])
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping headlines found!"


def test_stratification_proportions():
    train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
    df_train = pd.read_csv(train_path, encoding='utf-8-sig')
    df_test = pd.read_csv(test_path, encoding='utf-8-sig')

    train_props = df_train['Label'].value_counts(normalize=True)
    test_props = df_test['Label'].value_counts(normalize=True)

    for label in train_props.index:
        diff = abs(train_props[label] - test_props[label])
        assert diff < 0.005, f"Class {label} proportion difference too high: {diff:.4f}"
