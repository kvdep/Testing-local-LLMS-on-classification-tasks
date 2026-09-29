"""Модуль оркестрации распределения вычислений между несколькими GPU."""

import threading
from typing import Any, Dict, List, Optional
import pandas as pd
import torch

from .config import (
    HEAVY_MODEL_VRAM_THRESHOLD_GB,
    MODEL_REGISTRY,
)
from .model_loader import get_model_size_gb
from .worker import run_model_inference_pipeline


class BenchmarkOrchestrator:
    def __init__(
        self,
        experiments_base_dir: str,
        results_base_dir: str,
        force_rerun_dict: Optional[Dict[str, Any]] = None,
    ):
        self.experiments_base_dir = experiments_base_dir
        self.results_base_dir = results_base_dir
        self.force_rerun_dict = force_rerun_dict or {}
        self.git_lock = threading.Lock()

    def run_benchmark(
        self,
        models_to_run: List[str],
        df_test_shuffled: pd.DataFrame,
        device_mode: str = "auto",
    ) -> None:
        """
        Планирует и запускает эксперименты:
        Если доступно >= 2 GPU и выбран режим dual-gpu, распределяет легкие модели по потокам,
        а тяжелые запускает последовательно через device_map='auto'.
        """
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Если доступен 1 GPU или CPU
        if gpu_count < 2 or device_mode == "single-gpu":
            device = "cuda:0" if gpu_count > 0 else "cpu"
            print(f"[Orchestrator] Запуск последовательного инференса на {device}")
            run_model_inference_pipeline(
                device=device,
                models_to_run=models_to_run,
                df_test_shuffled=df_test_shuffled,
                experiments_base_dir=self.experiments_base_dir,
                results_base_dir=self.results_base_dir,
                force_rerun_dict=self.force_rerun_dict,
                lock=None,
            )
            return

        # Разделение моделей по весу
        sorted_models = []
        for m in models_to_run:
            if m in MODEL_REGISTRY:
                gb = get_model_size_gb(MODEL_REGISTRY[m].repo_id)
                sorted_models.append((m, gb))

        gpu0_queue = []
        gpu1_queue = []
        heavy_queue = []

        toggle = True
        for m_short, gb in sorted_models:
            # Granite принудительно оставляется в легкой очереди
            if gb >= HEAVY_MODEL_VRAM_THRESHOLD_GB and m_short != "granite_3_0_8b":
                heavy_queue.append(m_short)
            else:
                if toggle:
                    gpu0_queue.append(m_short)
                else:
                    gpu1_queue.append(m_short)
                toggle = not toggle

        print(f"[Orchestrator] Очередь GPU 0: {gpu0_queue}")
        print(f"[Orchestrator] Очередь GPU 1: {gpu1_queue}")
        print(f"[Orchestrator] Очередь Heavy (2x GPU auto): {heavy_queue}")

        threads = []
        if gpu0_queue:
            t0 = threading.Thread(
                target=run_model_inference_pipeline,
                kwargs={
                    "device": "cuda:0",
                    "models_to_run": gpu0_queue,
                    "df_test_shuffled": df_test_shuffled,
                    "experiments_base_dir": self.experiments_base_dir,
                    "results_base_dir": self.results_base_dir,
                    "force_rerun_dict": self.force_rerun_dict,
                    "lock": self.git_lock,
                },
            )
            threads.append(t0)
            t0.start()

        if gpu1_queue:
            t1 = threading.Thread(
                target=run_model_inference_pipeline,
                kwargs={
                    "device": "cuda:1",
                    "models_to_run": gpu1_queue,
                    "df_test_shuffled": df_test_shuffled,
                    "experiments_base_dir": self.experiments_base_dir,
                    "results_base_dir": self.results_base_dir,
                    "force_rerun_dict": self.force_rerun_dict,
                    "lock": self.git_lock,
                },
            )
            threads.append(t1)
            t1.start()

        for t in threads:
            t.join()

        print("[Orchestrator] Легкие и средние модели на GPU 0 и GPU 1 завершены.")

        # Последовательный запуск тяжелых моделей на 2 GPU
        if heavy_queue:
            print("[Orchestrator] Запуск тяжелого пула моделей на device_map='auto'...")
            run_model_inference_pipeline(
                device="auto",
                models_to_run=heavy_queue,
                df_test_shuffled=df_test_shuffled,
                experiments_base_dir=self.experiments_base_dir,
                results_base_dir=self.results_base_dir,
                force_rerun_dict=self.force_rerun_dict,
                lock=None,
            )
