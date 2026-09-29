"""Модуль генерации и адаптации структур промптов под различные архитектуры LLM."""

from typing import Any, Dict, List, Optional


def build_prompt_structure(
    mode: str,
    data_row: Optional[Dict[str, Any]] = None,
    batch_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Формирует унифицированный список сообщений (role, content) под заданную стратегию."""
    if mode == "zero_shot":
        return [
            {
                "role": "user",
                "content": (
                    "Analyze the following news headline and determine if it is clickbait. "
                    "Answer with 1 for clickbait or 0 for non-clickbait.\n"
                    f"Headline: {data_row['headline']}\nAnswer:"
                ),
            }
        ]

    elif mode == "zero_shot_structured":
        return [
            {
                "role": "user",
                "content": (
                    "Analyze the following news headline and return your answer strictly as a JSON object "
                    'with a single key "clickbait" mapping to either 1 or 0.\n'
                    f'Headline: {data_row["headline"]}\nJSON:'
                ),
            }
        ]

    elif mode == "zero_shot_limited":
        return [
            {
                "role": "user",
                "content": (
                    "Is the following headline clickbait? Answer strictly with a single character: 1 or 0. "
                    "Do not write anything else.\n"
                    f"Headline: {data_row['headline']}\nAnswer:"
                ),
            }
        ]

    elif mode == "system_prompt":
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert NLP classifier specialized in identifying clickbait headlines. "
                    "Respond with 1 if the headline is clickbait, and 0 if it is legitimate news."
                ),
            },
            {"role": "user", "content": f"Headline: {data_row['headline']}"},
        ]

    elif mode == "few_shot_system":
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert NLP classifier specialized in identifying clickbait headlines. "
                    "Respond with 1 if the headline is clickbait, and 0 if it is legitimate news."
                ),
            },
            {
                "role": "user",
                "content": "Headline: 10 Massive Secrets Celebrities Don't Want You To Know!",
            },
            {"role": "assistant", "content": "1"},
            {
                "role": "user",
                "content": "Headline: Scientists discover a new species of deep-sea jellyfish in the Pacific Ocean.",
            },
            {"role": "assistant", "content": "0"},
            {"role": "user", "content": f"Headline: {data_row['headline']}"},
        ]

    elif mode == "few_shot_structured":
        return [
            {
                "role": "system",
                "content": 'You are an expert NLP classifier. Respond strictly in valid JSON format: {"clickbait": X}',
            },
            {
                "role": "user",
                "content": "Headline: You won't believe what happened next to this poor puppy!",
            },
            {"role": "assistant", "content": '{"clickbait": 1}'},
            {
                "role": "user",
                "content": "The federal reserve raised interest rates by a quarter percent today.",
            },
            {"role": "assistant", "content": '{"clickbait": 0}'},
            {"role": "user", "content": f"Headline: {data_row['headline']}"},
        ]

    elif mode == "few_shot_limited":
        return [
            {
                "role": "user",
                "content": "Headline: 15 foods you should avoid at all costs\nAnswer:",
            },
            {"role": "assistant", "content": "1"},
            {
                "role": "user",
                "content": "Headline: European Union announces new carbon emissions targets for 2030\nAnswer:",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": f"Headline: {data_row['headline']}\nAnswer:",
            },
        ]

    elif mode == "zero_shot_multi":
        lines = [f"{i+1}. {r['headline']}" for i, r in enumerate(batch_rows)]
        txt = "\n".join(lines)
        return [
            {
                "role": "user",
                "content": (
                    "You are an expert NLP classifier. Analyze a batch of headlines and return a JSON object "
                    'with a single key "predictions" which maps to a list of integers (0 or 1) in the exact order of the inputs.\n'
                    f"Headlines:\n{txt}\nJSON:"
                ),
            }
        ]

    elif mode == "few_shot_multi":
        lines = [f"{i+1}. {r['headline']}" for i, r in enumerate(batch_rows)]
        txt = "\n".join(lines)
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert NLP classifier. Analyze a batch of headlines and return a JSON object "
                    'with a single key "predictions" containing a list of integers (0 or 1).'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Headlines:\n"
                    "1. This simple trick will save you thousands\n"
                    "2. French president arrives in Washington for official state visit"
                ),
            },
            {"role": "assistant", "content": '{"predictions": [1, 0]}'},
            {"role": "user", "content": f"Headlines:\n{txt}"},
        ]

    raise ValueError(f"Неизвестный режим формирования промпта: {mode}")


def inject_reasoning_constraint(
    p_struct: List[Dict[str, str]], model_name: str
) -> List[Dict[str, str]]:
    """
    Для рассуждающих моделей (DeepSeek-R1, Qwen) внедряет директиву ограничения бюджета рассуждений.
    Директива вставляется строго перед префиксами ответов JSON: или Answer:.
    """
    if not any(k in model_name.lower() for k in ["qwen", "deepseek"]):
        return p_struct

    modified = []
    for msg in p_struct:
        content = msg["content"]
        if "JSON:" in content:
            parts = content.split("JSON:")
            content = (
                parts[0]
                + "Note: Think extremely concisely (under 2-3 sentences max) before outputting the final JSON.\nJSON:"
                + "".join(parts[1:])
            )
        elif "Answer:" in content:
            parts = content.split("Answer:")
            content = (
                parts[0]
                + "Note: Think extremely concisely (under 2-3 sentences max) before outputting the final Answer.\nAnswer:"
                + "".join(parts[1:])
            )
        else:
            content += " Note: Your thinking/reasoning phase must be extremely concise (under 2-3 sentences max)."
        modified.append({"role": msg["role"], "content": content})
    return modified


def merge_system_prompt_to_user(
    p_struct: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Для моделей без поддержки роли 'system' (например, Gemma 2),
    объединяет инструкции system в первое пользовательское сообщение.
    """
    if not isinstance(p_struct, list):
        return p_struct

    has_system = any(msg["role"] == "system" for msg in p_struct)
    if not has_system:
        return p_struct

    system_content = ""
    new_struct = []
    for msg in p_struct:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            if msg["role"] == "user" and system_content:
                new_struct.append(
                    {
                        "role": "user",
                        "content": f"Instructions:\n{system_content}\n\nInput Headline:\n{msg['content']}",
                    }
                )
                system_content = ""
            else:
                new_struct.append(msg)
    return new_struct


def format_prompt_for_processor(
    p_struct: List[Dict[str, Any]], tokenizer_or_processor: Any
) -> List[Dict[str, Any]]:
    """
    Форматирует текстовый ввод под требования AutoProcessor (Gemma 3, Gemma 4).
    Преобразует строковый content в структуру [{'type': 'text', 'text': ...}].
    """
    # Проверка, является ли объект экземпляром AutoProcessor
    class_name = tokenizer_or_processor.__class__.__name__
    if "Processor" in class_name and isinstance(p_struct, list):
        formatted = []
        for msg in p_struct:
            c = msg["content"]
            formatted.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": c}] if isinstance(c, str) else c,
                }
            )
        return formatted
    return p_struct
