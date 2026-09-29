"""Базовый классический пайплайн машинного обучения: Feature Engineering + Logistic Regression."""

import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_flesch_reading_ease(text: str) -> float:
    """Вычисляет индекс удобочитаемости Флеша: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)."""
    words = re.findall(r"\b\w+\b", text)
    num_words = max(len(words), 1)
    num_sentences = max(len(re.split(r"[.!?]+", text)), 1)

    # Приближенный подсчет слогов по гласным
    vowels = "aeiouyAEIOUY"
    num_syllables = sum(1 for char in text if char in vowels)
    num_syllables = max(num_syllables, 1)

    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return float(score)


def calculate_flesch_kincaid_grade(text: str) -> float:
    """Вычисляет индекс удобочитаемости Флеша-Кинкейда: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59."""
    words = re.findall(r"\b\w+\b", text)
    num_words = max(len(words), 1)
    num_sentences = max(len(re.split(r"[.!?]+", text)), 1)

    vowels = "aeiouyAEIOUY"
    num_syllables = sum(1 for char in text if char in vowels)
    num_syllables = max(num_syllables, 1)

    score = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return float(score)


def extract_heuristic_features(df: pd.DataFrame, text_col: str = "headline") -> np.ndarray:
    """Извлекает числовые эвристические признаки из заголовков."""
    features = []
    for text in df[text_col]:
        text_str = str(text)
        length_chars = len(text_str)
        words = text_str.split()
        length_words = len(words)

        upper_chars = sum(1 for c in text_str if c.isupper())
        upper_char_ratio = upper_chars / max(length_chars, 1)

        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        upper_word_ratio = upper_words / max(length_words, 1)

        digits_count = sum(1 for c in text_str if c.isdigit())
        exclamations_count = text_str.count("!")
        questions_count = text_str.count("?")

        flesch_ease = calculate_flesch_reading_ease(text_str)
        flesch_grade = calculate_flesch_kincaid_grade(text_str)

        features.append(
            [
                length_chars,
                length_words,
                upper_char_ratio,
                upper_word_ratio,
                digits_count,
                exclamations_count,
                questions_count,
                flesch_ease,
                flesch_grade,
            ]
        )
    return np.array(features)


class ClassicalClickbaitClassifier:
    """Классификатор на основе LogisticRegression и комбинированного TF-IDF + эвристик."""

    def __init__(self, max_word_features: int = 10000, max_char_features: int = 10000, c: float = 1.0):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_word_features)
        self.char_vec = TfidfVectorizer(ngram_range=(2, 5), analyzer="char", max_features=max_char_features)
        self.model = LogisticRegression(C=c, max_iter=1000, random_state=42)

    def fit(self, df_train: pd.DataFrame, text_col: str = "headline", label_col: str = "clickbait"):
        texts = df_train[text_col].fillna("").astype(str)
        y = df_train[label_col].values

        x_word = self.word_vec.fit_transform(texts)
        x_char = self.char_vec.fit_transform(texts)
        x_dense = extract_heuristic_features(df_train, text_col=text_col)

        x_all = hstack([x_word, x_char, x_dense])
        self.model.fit(x_all, y)
        return self

    def predict(self, df: pd.DataFrame, text_col: str = "headline") -> np.ndarray:
        texts = df[text_col].fillna("").astype(str)
        x_word = self.word_vec.transform(texts)
        x_char = self.char_vec.transform(texts)
        x_dense = extract_heuristic_features(df, text_col=text_col)

        x_all = hstack([x_word, x_char, x_dense])
        return self.model.predict(x_all)

    def evaluate(
        self, df_test: pd.DataFrame, text_col: str = "headline", label_col: str = "clickbait"
    ) -> Dict[str, float]:
        y_true = df_test[label_col].values
        y_pred = self.predict(df_test, text_col=text_col)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
        }
