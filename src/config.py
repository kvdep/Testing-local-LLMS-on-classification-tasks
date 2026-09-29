"""Конфигурация бенчмарка локальных LLM и вспомогательных моделей."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    repo_id: str
    params_b: float
    fallback_size_gb: float
    use_processor: bool = False
    requires_quantization: bool = False
    default_dtype: torch.dtype = torch.float16
    sample_limit: int = 50
    batch_size: int = 10


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "deepseek_r1_1_5b": ModelSpec(
        short_name="deepseek_r1_1_5b",
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        params_b=1.5,
        fallback_size_gb=3.0,
        sample_limit=50,
        batch_size=10,
    ),
    "stablelm2_1_6b": ModelSpec(
        short_name="stablelm2_1_6b",
        repo_id="stabilityai/stablelm-2-zephyr-1_6b",
        params_b=1.6,
        fallback_size_gb=3.2,
        sample_limit=50,
        batch_size=10,
    ),
    "smollm2_1_3b": ModelSpec(
        short_name="smollm2_1_3b",
        repo_id="HuggingFaceTB/SmolLM2-1.3B-Instruct",
        params_b=1.3,
        fallback_size_gb=2.6,
        sample_limit=50,
        batch_size=10,
    ),
    "gemma_4_e2b": ModelSpec(
        short_name="gemma_4_e2b",
        repo_id="google/gemma-4-E2B-it",
        params_b=2.3,
        fallback_size_gb=5.0,
        use_processor=True,
        requires_quantization=True,
        default_dtype=torch.bfloat16,
        sample_limit=20,
        batch_size=5,
    ),
    "llama_3_2_3b": ModelSpec(
        short_name="llama_3_2_3b",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.2,
        fallback_size_gb=6.0,
        sample_limit=20,
        batch_size=5,
    ),
    "smollm3_3b": ModelSpec(
        short_name="smollm3_3b",
        repo_id="HuggingFaceTB/SmolLM3-3B",
        params_b=3.0,
        fallback_size_gb=6.0,
        sample_limit=20,
        batch_size=5,
    ),
    "phi_4_mini": ModelSpec(
        short_name="phi_4_mini",
        repo_id="microsoft/Phi-4-mini-instruct",
        params_b=3.8,
        fallback_size_gb=7.6,
        sample_limit=20,
        batch_size=5,
    ),
    "gemma_3_4b": ModelSpec(
        short_name="gemma_3_4b",
        repo_id="google/gemma-3-4b-it",
        params_b=4.0,
        fallback_size_gb=8.0,
        use_processor=True,
        requires_quantization=True,
        default_dtype=torch.bfloat16,
        sample_limit=20,
        batch_size=5,
    ),
    "gemma_4_e4b": ModelSpec(
        short_name="gemma_4_e4b",
        repo_id="google/gemma-4-E4B-it",
        params_b=4.5,
        fallback_size_gb=9.5,
        use_processor=True,
        requires_quantization=True,
        default_dtype=torch.bfloat16,
        sample_limit=10,
        batch_size=2,
    ),
    "mistral_7b": ModelSpec(
        short_name="mistral_7b",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        params_b=7.0,
        fallback_size_gb=14.5,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
    "granite_3_0_8b": ModelSpec(
        short_name="granite_3_0_8b",
        repo_id="ibm-granite/granite-3.0-8b-instruct",
        params_b=8.0,
        fallback_size_gb=16.0,
        requires_quantization=True,
        sample_limit=50,
        batch_size=10,
    ),
    "qwen_3_8b": ModelSpec(
        short_name="qwen_3_8b",
        repo_id="Qwen/Qwen3-8B",
        params_b=8.0,
        fallback_size_gb=16.0,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
    "gemma_2_9b": ModelSpec(
        short_name="gemma_2_9b",
        repo_id="google/gemma-2-9b-it",
        params_b=9.0,
        fallback_size_gb=18.0,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
    "yi_1_5_9b": ModelSpec(
        short_name="yi_1_5_9b",
        repo_id="01-ai/Yi-1.5-9B-Chat",
        params_b=9.0,
        fallback_size_gb=18.0,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
    "stablelm2_12b": ModelSpec(
        short_name="stablelm2_12b",
        repo_id="stabilityai/stablelm-2-12b-chat",
        params_b=12.0,
        fallback_size_gb=24.0,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
    "codestral_22b": ModelSpec(
        short_name="codestral_22b",
        repo_id="mistralai/Codestral-22B-v0.1",
        params_b=22.0,
        fallback_size_gb=44.0,
        requires_quantization=True,
        sample_limit=10,
        batch_size=2,
    ),
}

EXPERIMENT_TYPES: List[str] = [
    "zero_shot",
    "zero_shot_structured",
    "zero_shot_limited",
    "system_prompt",
    "few_shot_system",
    "few_shot_structured",
    "few_shot_limited",
    "zero_shot_multioutput_structured",
    "few_shot_multioutput_structured",
]

HEAVY_MODEL_VRAM_THRESHOLD_GB: float = 13.0
RANDOM_SEED: int = 42
MAX_NEW_TOKENS: int = 512
