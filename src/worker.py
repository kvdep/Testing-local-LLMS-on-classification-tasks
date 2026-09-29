"""Модуль изолированного воркера инференса на выделенном GPU."""

import json
import os
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import torch

from .config import MODEL_REGISTRY, EXPERIMENT_TYPES, MAX_NEW_TOKENS
from .metrics import compute_classification_metrics, update_summary_csv
from .model_loader import (
    clean_vram,
    delete_from_cache,
    load_model_and_tokenizer,
)
from .parsers import extract_label_multi, extract_label_single
from .prompts import (
    build_prompt_structure,
    format_prompt_for_processor,
    inject_reasoning_constraint,
    merge_system_prompt_to_user,
)


def run_model_inference_pipeline(
    device: str,
    models_to_run: List[str],
    df_test_shuffled: pd.DataFrame,
    experiments_base_dir: str,
    results_base_dir: str,
    force_rerun_dict: Optional[Dict[str, Any]] = None,
    lock: Optional[Any] = None,
) -> None:
    """Выполняет инференс списка моделей на указанном устройстве (GPU или auto)."""
    force_rerun = force_rerun_dict or {}

    for m_short in models_to_run:
        if m_short not in MODEL_REGISTRY:
            continue
        spec = MODEL_REGISTRY[m_short]
        m_folder = os.path.join(
            experiments_base_dir, m_short.replace("/", "_").replace("-", "_")
        )
        os.makedirs(m_folder, exist_ok=True)

        test_limit = spec.sample_limit
        m_batch_size = spec.batch_size

        df_test = df_test_shuffled.head(test_limit)
        y_true = df_test["clickbait"].tolist()

        # Проверка кэшированных предсказаний
        needed_exps = []
        cached_predictions = {}

        for exp in EXPERIMENT_TYPES:
            exp_folder = os.path.join(m_folder, exp)
            predictions_path = os.path.join(exp_folder, "predictions.csv")

            existing_cache = {}
            if os.path.exists(predictions_path):
                try:
                    df_existing = pd.read_csv(predictions_path)
                    if (
                        "prediction" in df_existing.columns
                        and "raw_response" in df_existing.columns
                    ):
                        for _, row in df_existing.iterrows():
                            h = row["headline"]
                            p = row["prediction"]
                            r = row["raw_response"]
                            if (
                                pd.notna(h)
                                and pd.notna(p)
                                and pd.notna(r)
                                and str(r).strip() != ""
                            ):
                                existing_cache[str(h).strip()] = (
                                    int(p),
                                    str(r),
                                )
                except Exception:
                    pass

            missing_indices = []
            for idx, row in df_test.iterrows():
                h = str(row["headline"]).strip()
                if h not in existing_cache:
                    missing_indices.append(idx)

            cached_predictions[exp] = (existing_cache, missing_indices)

            # Проверка флага принудительного перезапуска
            is_forced = False
            if m_short in force_rerun:
                cfg = force_rerun[m_short]
                if cfg == "all" or (isinstance(cfg, list) and exp in cfg):
                    is_forced = True

            if len(missing_indices) > 0 or is_forced:
                needed_exps.append(exp)

        if not needed_exps:
            continue

        # Загрузка модели на устройство
        try:
            model, tk = load_model_and_tokenizer(spec, device=device)
        except Exception as e:
            print(f"[{device}] Ошибка загрузки {spec.repo_id}: {e}")
            continue

        for exp in EXPERIMENT_TYPES:
            exp_folder = os.path.join(m_folder, exp)
            os.makedirs(exp_folder, exist_ok=True)
            predictions_path = os.path.join(exp_folder, "predictions.csv")
            metrics_path = os.path.join(exp_folder, "metrics.json")

            existing_cache, missing_indices = cached_predictions[exp]
            if m_short in force_rerun:
                cfg = force_rerun[m_short]
                if cfg == "all" or (isinstance(cfg, list) and exp in cfg):
                    existing_cache = {}
                    missing_indices = list(df_test.index)

            # Если все данные уже в кэше — обновляем метрики и идем дальше
            if len(missing_indices) == 0:
                preds = [
                    existing_cache[str(row["headline"]).strip()][0]
                    for _, row in df_test.iterrows()
                ]
                raw_resps = [
                    existing_cache[str(row["headline"]).strip()][1]
                    for _, row in df_test.iterrows()
                ]

                df_out = df_test.copy()
                df_out["prediction"] = preds
                df_out["raw_response"] = raw_resps
                df_out.to_csv(predictions_path, index=False)

                m_metrics = compute_classification_metrics(y_true, preds)
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(m_metrics, f, indent=4)

                if lock:
                    with lock:
                        update_summary_csv(
                            results_base_dir, exp, m_short, m_metrics
                        )
                else:
                    update_summary_csv(
                        results_base_dir, exp, m_short, m_metrics
                    )
                continue

            df_missing = df_test.loc[missing_indices]

            if "multioutput" in exp:
                mode_key = (
                    "zero_shot_multi" if "zero" in exp else "few_shot_multi"
                )
                missing_records = df_missing.to_dict(orient="records")
                for idx in range(0, len(missing_records), m_batch_size):
                    b_rows = missing_records[idx : idx + m_batch_size]
                    p_struct = build_prompt_structure(mode_key, batch_rows=b_rows)
                    p_struct = inject_reasoning_constraint(p_struct, m_short)

                    if "gemma_2" in m_short.lower():
                        p_struct = merge_system_prompt_to_user(p_struct)

                    p_struct = format_prompt_for_processor(p_struct, tk)
                    tmpl_kwargs = {}
                    if "gemma_4" in m_short.lower() or "gemma_3" in m_short.lower():
                        tmpl_kwargs["enable_thinking"] = False

                    # Самовосстанавливающаяся компиляция шаблона
                    try:
                        in_txt = (
                            tk.apply_chat_template(
                                p_struct,
                                tokenize=False,
                                add_generation_prompt=True,
                                **tmpl_kwargs,
                            )
                            if isinstance(p_struct, list)
                            else p_struct
                        )
                    except Exception as e_tmpl:
                        if "system" in str(e_tmpl).lower():
                            p_struct = merge_system_prompt_to_user(p_struct)
                            p_struct_formatted = format_prompt_for_processor(
                                p_struct, tk
                            )
                            in_txt = tk.apply_chat_template(
                                p_struct_formatted,
                                tokenize=False,
                                add_generation_prompt=True,
                                **tmpl_kwargs,
                            )
                        else:
                            raise e_tmpl

                    ids_raw = tk(text=in_txt, return_tensors="pt")
                    model_device = next(model.parameters()).device
                    ids = {
                        k: v.to(model_device)
                        for k, v in ids_raw.items()
                        if isinstance(v, torch.Tensor)
                    }

                    with torch.no_grad():
                        out_ids = model.generate(
                            **ids,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            use_cache=True,
                        )

                    skip_special = not (
                        "gemma_4" in m_short.lower()
                        or "gemma_3" in m_short.lower()
                    )
                    gen_text = tk.decode(
                        out_ids[0][ids["input_ids"].shape[-1] :],
                        skip_special_tokens=skip_special,
                    )

                    if "gemma_4" in m_short.lower() or "gemma_3" in m_short.lower():
                        try:
                            parsed = tk.parse_response(gen_text)
                            gen_text = parsed.get("content", gen_text)
                        except Exception:
                            pass

                    extracted_batch = extract_label_multi(gen_text, len(b_rows))
                    for r_idx, r_item in enumerate(b_rows):
                        existing_cache[str(r_item["headline"]).strip()] = (
                            extracted_batch[r_idx],
                            gen_text,
                        )
            else:
                for idx, row in df_missing.iterrows():
                    p_struct = build_prompt_structure(exp, data_row=row)
                    p_struct = inject_reasoning_constraint(p_struct, m_short)

                    if "gemma_2" in m_short.lower():
                        p_struct = merge_system_prompt_to_user(p_struct)

                    p_struct = format_prompt_for_processor(p_struct, tk)
                    tmpl_kwargs = {}
                    if "gemma_4" in m_short.lower() or "gemma_3" in m_short.lower():
                        tmpl_kwargs["enable_thinking"] = False

                    try:
                        in_txt = (
                            tk.apply_chat_template(
                                p_struct,
                                tokenize=False,
                                add_generation_prompt=True,
                                **tmpl_kwargs,
                            )
                            if isinstance(p_struct, list)
                            else p_struct
                        )
                    except Exception as e_tmpl:
                        if "system" in str(e_tmpl).lower():
                            p_struct = merge_system_prompt_to_user(p_struct)
                            p_struct_formatted = format_prompt_for_processor(
                                p_struct, tk
                            )
                            in_txt = tk.apply_chat_template(
                                p_struct_formatted,
                                tokenize=False,
                                add_generation_prompt=True,
                                **tmpl_kwargs,
                            )
                        else:
                            raise e_tmpl

                    ids_raw = tk(text=in_txt, return_tensors="pt")
                    model_device = next(model.parameters()).device
                    ids = {
                        k: v.to(model_device)
                        for k, v in ids_raw.items()
                        if isinstance(v, torch.Tensor)
                    }

                    with torch.no_grad():
                        out_ids = model.generate(
                            **ids,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            use_cache=True,
                        )

                    skip_special = not (
                        "gemma_4" in m_short.lower()
                        or "gemma_3" in m_short.lower()
                    )
                    gen_text = tk.decode(
                        out_ids[0][ids["input_ids"].shape[-1] :],
                        skip_special_tokens=skip_special,
                    )

                    if "gemma_4" in m_short.lower() or "gemma_3" in m_short.lower():
                        try:
                            parsed = tk.parse_response(gen_text)
                            gen_text = parsed.get("content", gen_text)
                        except Exception:
                            pass

                    lbl = extract_label_single(gen_text)
                    existing_cache[str(row["headline"]).strip()] = (
                        lbl,
                        gen_text,
                    )

            preds = []
            raw_resps = []
            for _, row in df_test.iterrows():
                h = str(row["headline"]).strip()
                pred, raw_resp = existing_cache[h]
                preds.append(pred)
                raw_resps.append(raw_resp)

            df_out = df_test.copy()
            df_out["prediction"] = preds
            df_out["raw_response"] = raw_resps
            df_out.to_csv(predictions_path, index=False)

            m_metrics = compute_classification_metrics(y_true, preds)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(m_metrics, f, indent=4)

            if lock:
                with lock:
                    update_summary_csv(results_base_dir, exp, m_short, m_metrics)
            else:
                update_summary_csv(results_base_dir, exp, m_short, m_metrics)

        del model
        del tk
        clean_vram()
        delete_from_cache(spec.repo_id)
