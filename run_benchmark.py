"""Основной интерфейс командной строки для запуска бенчмарка локальных LLM и базовых моделей."""

import argparse
import os
import sys
import pandas as pd

from src.config import EXPERIMENT_TYPES, MODEL_REGISTRY, RANDOM_SEED
from src.orchestrator import BenchmarkOrchestrator
from src.baselines.classical_ml import ClassicalClickbaitClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Бенчмарк локальных LLM на задаче классификации кликбейта"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="Список моделей для тестирования (по умолчанию: все)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=EXPERIMENT_TYPES,
        help="Список стратегий промптинга",
    )
    parser.add_argument(
        "--device-mode",
        choices=["auto", "dual-gpu", "single-gpu"],
        default="auto",
        help="Режим распределения по видеокартам",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/test.csv",
        help="Путь к тестовому CSV файлу",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Папка сохранения предсказаний",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Папка сохранения итоговых сводок метрик",
    )
    parser.add_argument(
        "--force-rerun",
        type=str,
        default=None,
        help="Имя модели для принудительного перезапуска экспериментов",
    )
    parser.add_argument(
        "--run-classical-baseline",
        action="store_true",
        help="Запустить обучение и оценку классической модели Logistic Regression",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/train.csv",
        help="Путь к обучающему CSV файлу для baseline",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_classical_baseline:
        print("[Baseline] Запуск классического ML...")
        if not os.path.exists(args.train_data_path) or not os.path.exists(args.data_path):
            print(f"Ошибка: Не найдены файлы датасета {args.train_data_path} или {args.data_path}")
            sys.exit(1)
        df_train = pd.read_csv(args.train_data_path)
        df_test = pd.read_csv(args.data_path)
        clf = ClassicalClickbaitClassifier()
        clf.fit(df_train)
        metrics = clf.evaluate(df_test)
        print("Результаты Classical ML Baseline:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
        return

    if not os.path.exists(args.data_path):
        print(f"[Warning] Файл {args.data_path} не найден.")
        print("Используется тестовый срез из существующих экспериментов для демонстрации структуры.")
        # Создаем демонстрационный датасет из имеющихся файлов predictions
        sample_path = os.path.join(args.experiments_dir, "deepseek_r1_1_5b", "zero_shot", "predictions.csv")
        if os.path.exists(sample_path):
            df_test = pd.read_csv(sample_path)[["headline", "clickbait"]]
        else:
            print("Ошибка: Тестовые данные отсутствуют.")
            sys.exit(1)
    else:
        df_test = pd.read_csv(args.data_path)

    df_test_shuffled = df_test.sample(n=len(df_test), random_state=RANDOM_SEED).reset_index(drop=True)

    force_rerun_dict = {}
    if args.force_rerun:
        force_rerun_dict[args.force_rerun] = "all"

    orchestrator = BenchmarkOrchestrator(
        experiments_base_dir=args.experiments_dir,
        results_base_dir=args.results_dir,
        force_rerun_dict=force_rerun_dict,
    )

    print(f"[Main] Запуск бенчмарка для {len(args.models)} моделей...")
    orchestrator.run_benchmark(
        models_to_run=args.models,
        df_test_shuffled=df_test_shuffled,
        device_mode=args.device_mode,
    )
    print("[Main] Бенчмарк завершен.")


if __name__ == "__main__":
    main()
