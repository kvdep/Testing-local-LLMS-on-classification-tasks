"""Базовый пайплайн дообучения энкодера DistilBERT с атрибуцией признаков (Explainable AI / XAI)."""

from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DistilBertClickbaitClassifier:
    """Обертка над distilbert-base-uncased для классификации кликбейта и анализа важности токенов."""

    def __init__(self, model_id: str = "distilbert/distilbert-base-uncased", num_labels: int = 2):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }

    def explain_prediction(self, text: str) -> List[Tuple[str, float]]:
        """
        Вычисляет атрибуцию важности токенов методом Integrated Gradients / градиентов входных эмбеддингов.
        Возвращает пары (токен, показатель_важности).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        embedding_layer = self.model.distilbert.embeddings.word_embeddings
        input_ids = inputs["input_ids"]
        input_embeds = embedding_layer(input_ids).detach()
        input_embeds.requires_grad = True

        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        pred_class = logits.argmax(dim=-1).item()

        # Расчет градиента предсказанного класса по эмбеддингам
        logits[0, pred_class].backward()
        grads = input_embeds.grad[0]  # [seq_len, hidden_dim]

        # Важность как норма градиента, взвешенная входным эмбеддингом
        attribution = (grads * input_embeds[0]).sum(dim=-1).detach().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return list(zip(tokens, [float(a) for a in attribution]))
