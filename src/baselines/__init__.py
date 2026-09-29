"""Модули базовых решений (классический ML, дистиллированный трансформер, QLoRA)."""

from .classical_ml import ClassicalClickbaitClassifier
from .distilbert_baseline import DistilBertClickbaitClassifier
from .qlora_llm import setup_qlora_model

__all__ = [
    "ClassicalClickbaitClassifier",
    "DistilBertClickbaitClassifier",
    "setup_qlora_model",
]
