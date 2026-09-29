"""Модуль загрузки моделей, токенизаторов, управления VRAM и дисковым кэшем."""

import gc
import os
import shutil
import time
from typing import Any, Tuple
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from .config import ModelSpec, MODEL_REGISTRY


def get_model_size_gb(repo_id: str, token: str = None) -> float:
    """Определяет физический объем весов модели на HuggingFace Hub через files_metadata."""
    try:
        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        info = api.model_info(repo_id=repo_id, files_metadata=True)
        files = [
            f
            for f in info.siblings
            if f.size
            and (f.rfilename.endswith(".safetensors") or f.rfilename.endswith(".bin"))
        ]
        safetensors = [f for f in files if f.rfilename.endswith(".safetensors")]
        target_files = safetensors if safetensors else files
        bytes_sum = sum(f.size for f in target_files)
        return bytes_sum / (1024**3)
    except Exception:
        # Резервные значения из реестра моделей
        for spec in MODEL_REGISTRY.values():
            if spec.repo_id == repo_id:
                return spec.fallback_size_gb
        return 8.0


def clean_vram() -> None:
    """Выполняет сборку мусора Python и очищает кэш аллокатора CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)


def delete_from_cache(repo_id: str) -> None:
    """Удаляет загруженные веса модели из локального дискового кэша Hugging Face Hub."""
    folder_name = f"models--{repo_id.replace('/', '--')}"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    target_path = os.path.join(cache_dir, folder_name)
    if os.path.exists(target_path):
        try:
            shutil.rmtree(target_path)
        except Exception:
            pass


def load_model_and_tokenizer(
    spec: ModelSpec,
    device: str,
    token: str = None,
) -> Tuple[Any, Any]:
    """
    Загружает модель и токенизатор/процессор с учетом требований квантования и архитектуры.
    Для Gemma 3/4 принудительно выставляет compute_dtype=torch.bfloat16.
    """
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HF_KEY")
    repo_id = spec.repo_id

    # Настройка квантования BitsAndBytes
    q_cfg = None
    if spec.requires_quantization:
        q_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=spec.default_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # Загрузка токенизатора или процессора
    if spec.use_processor:
        tk = AutoProcessor.from_pretrained(repo_id, token=hf_token)
        try:
            chat_tmpl_path = hf_hub_download(
                repo_id, "chat_template.jinja", token=hf_token
            )
            with open(chat_tmpl_path, "r", encoding="utf-8") as f:
                template_string = f.read()
                if hasattr(tk, "tokenizer"):
                    tk.tokenizer.chat_template = template_string
                else:
                    tk.chat_template = template_string
        except Exception:
            pass

        if hasattr(tk, "tokenizer") and tk.tokenizer.pad_token is None:
            tk.tokenizer.pad_token = tk.tokenizer.eos_token
        elif hasattr(tk, "pad_token") and tk.pad_token is None:
            tk.pad_token = tk.eos_token

        if q_cfg:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                quantization_config=q_cfg,
                device_map=device,
                token=hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=spec.default_dtype,
                device_map=device,
                token=hf_token,
            )
    else:
        tk = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True, token=hf_token
        )
        if tk.pad_token is None:
            tk.pad_token = tk.eos_token

        if q_cfg:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                quantization_config=q_cfg,
                device_map=device,
                trust_remote_code=True,
                token=hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=spec.default_dtype,
                device_map=device,
                trust_remote_code=True,
                token=hf_token,
            )

    return model, tk
