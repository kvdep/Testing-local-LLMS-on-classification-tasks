"""Пакет бенчмарка локальных LLM на задачах бинарной классификации текста."""

from .config import MODEL_REGISTRY, EXPERIMENT_TYPES, ModelSpec
from .prompts import build_prompt_structure, inject_reasoning_constraint, merge_system_prompt_to_user
from .parsers import clean_generation_text, extract_label_single, extract_label_multi
from .metrics import compute_classification_metrics, update_summary_csv
from .model_loader import load_model_and_tokenizer, clean_vram, delete_from_cache, get_model_size_gb
from .worker import run_model_inference_pipeline
from .orchestrator import BenchmarkOrchestrator

__all__ = [
    "MODEL_REGISTRY",
    "EXPERIMENT_TYPES",
    "ModelSpec",
    "build_prompt_structure",
    "inject_reasoning_constraint",
    "merge_system_prompt_to_user",
    "clean_generation_text",
    "extract_label_single",
    "extract_label_multi",
    "compute_classification_metrics",
    "update_summary_csv",
    "load_model_and_tokenizer",
    "clean_vram",
    "delete_from_cache",
    "get_model_size_gb",
    "run_model_inference_pipeline",
    "BenchmarkOrchestrator",
]
