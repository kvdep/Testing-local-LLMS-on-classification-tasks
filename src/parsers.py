"""Модуль очистки генераций и извлечения меток классификации."""

import json
import re
from typing import List


def clean_generation_text(text: str) -> str:
    """Удаляет из текста вывода служебные токены, теги размышлений и технические артефакты."""
    # Очистка скрытых каналов рассуждений
    text = re.sub(r"<\|channel\|>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[THINK\].*?\[/THINK\]", "", text, flags=re.DOTALL)

    # Список служебных токенов различных семейств LLM
    control_patterns = [
        r"<\|turn\|>",
        r"<\|turn>",
        r"<turn\|>",
        r"<\|model\|>",
        r"<\|user\|>",
        r"<bos>",
        r"<eos>",
        r"assistant\n",
        r"user\n",
        r"system\n",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[UTTERANCE\]",
        r"\[/UTTERANCE\]",
    ]
    for pattern in control_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def extract_label_single(raw_response: str) -> int:
    """
    Многоуровневый парсер для извлечения бинарной метки (0 или 1) из одиночного ответа модели:
    Уровень 1: Поиск валидного JSON объекта {"clickbait": 0/1}
    Уровень 2: Поиск явных текстовых ключей (answer: X, prediction: X, label: X, clickbait: X)
    Уровень 3: Фильтрация артефактов инструкции и поиск изолированных цифр
    Уровень 4: Обратное сканирование последнего вхождения 0 или 1
    Fallback: 0
    """
    clean_resp = clean_generation_text(raw_response)

    # Уровень 1: Строгий JSON
    match_json = re.search(r'\{\s*"clickbait"\s*:\s*([01])\s*\}', clean_resp)
    if match_json:
        return int(match_json.group(1))

    # Уровень 2: Явные маркеры ответа
    match_explicit = re.search(
        r"(?:answer|prediction|label|clickbait)\s*:\s*([01])", clean_resp, re.IGNORECASE
    )
    if match_explicit:
        return int(match_explicit.group(1))

    # Уровень 3: Удаление текстовых формулировок из промпта, которые содержат 0 и 1
    normalized = re.sub(r"\b1\s+or\s+0\b|\b0\s+or\s+1\b", "", clean_resp, flags=re.IGNORECASE)
    normalized = re.sub(r"\b1\s+for\s+clickbait\b|\b0\s+for\s+non-clickbait\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b1\s+if\s+the\s+headline\s+is\s+clickbait\b|\b0\s+if\s+it\s+is\s+legitimate\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    digits = re.findall(r"\b([01])\b", normalized)
    if digits:
        return int(digits[0])

    # Уровень 4: Последний встреченный символ класса
    for char in reversed(clean_resp):
        if char in ("0", "1"):
            return int(char)

    return 0


def extract_label_multi(raw_response: str, expected_count: int) -> List[int]:
    """
    Извлекает список меток (0/1) заданной длины из батчевого ответа модели:
    Уровень 1: JSON объект {"predictions": [0, 1, ...]}
    Уровень 2: Регулярное выражение для всех изолированных цифр 0/1
    Дополняет нулями в случае недостаточного количества элементов.
    """
    clean_resp = clean_generation_text(raw_response)

    # Уровень 1: Структурированный JSON массив
    try:
        match_json = re.search(r'\{\s*"predictions"\s*:\s*(\[[^\]]*\])\s*\}', clean_resp)
        if match_json:
            vals = json.loads(match_json.group(1))
            if isinstance(vals, list):
                result = [int(x) if str(x) in ("0", "1") else 0 for x in vals[:expected_count]]
                if len(result) < expected_count:
                    result += [0] * (expected_count - len(result))
                return result
    except Exception:
        pass

    # Уровень 2: Извлечение всех цифр 0 и 1 из текста
    digits = [int(x) for x in re.findall(r"\b([01])\b", clean_resp)]
    if len(digits) < expected_count:
        digits += [0] * (expected_count - len(digits))

    return digits[:expected_count]
