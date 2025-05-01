from yandex_cloud_ml_sdk import YCloudML
from pathlib import Path
import json
import os
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import time

lang = "ru"
# lang = "en"
# lang = "chs"
load_dotenv()

class YandexRAGEvaluator:
    def __init__(self, folder_id: str, api_key: str, assistant_name: str = "test_mai_admissions8"):
        print("\nИнициализация YandexRAGEvaluator...")
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        
        # Находим существующего ассистента
        print("Поиск ассистента...")
        assistants = [a for a in self.sdk.assistants.list() if a.name == assistant_name]
        if not assistants:
            print("\nОшибка: Ассистент не найден!")
            print("Возможные причины:")
            print("1. Ассистент еще не создан")
            print("2. Неправильное имя ассистента")
            print("3. Проблемы с доступом к API")
            print("\nДоступные ассистенты:")
            for a in self.sdk.assistants.list():
                print(f"- {a.name} (ID: {a.id})")
            raise ValueError(f"Ассистент с именем {assistant_name} не найден")
        
        self.assistant = assistants[0]
        print(f"Ассистент найден: {self.assistant.id}")
        
    def evaluate_rag_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Полная оценка RAG-системы"""
        results = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_relevance": [],
            "answer_correctness": []
        }
        
        for item in tqdm(test_data, desc="Evaluating RAG system"):
            # Создаем новый thread для каждого теста
            thread = self.sdk.threads.create(ttl_days=1, expiration_policy="static")
            try:
                print(f"\nОбработка вопроса: {item['question']}")
                
                # Отправляем вопрос
                print("Отправка вопроса...")
                thread.write(item["context"])
                
                # Запускаем ассистента
                print("Запуск ассистента...")
                run = self.assistant.run(thread)
                
                # Ждем ответ
                print("Ожидание ответа...")
                response = run.wait()
                
                if response is None:
                    print("ОШИБКА: Не получен ответ от модели")
                    continue
                    
                if not hasattr(response, 'text') or response.text is None:
                    print("ОШИБКА: Ответ не содержит текста")
                    print(f"Тип ответа: {type(response)}")
                    print(f"Атрибуты ответа: {dir(response)}")
                    continue
                
                # Получаем ответ
                answer = response.text.strip()
                print(f"Получен ответ длиной {len(answer)} символов")
                print(f"Ответ: {answer}")
                
                # Оцениваем метрики
                print("Оценка faithfulness...")
                faithfulness = self._evaluate_metric(
                    "faithfulness",
                    item["question"],
                    answer,
                    item["context"]
                )
                
                print("Оценка answer_relevance...")
                answer_relevance = self._evaluate_metric(
                    "answer_relevance",
                    item["question"],
                    answer
                )
                
                print("Оценка context_relevance...")
                context_relevance = self._evaluate_metric(
                    "context_relevance",
                    item["question"],
                    item["context"]
                )
                
                print("Оценка answer_correctness...")
                answer_correctness = self._evaluate_metric(
                    "answer_correctness",
                    answer,
                    item["ground_truth"]
                )
                
                # Сохраняем результаты
                results["faithfulness"].append(faithfulness)
                results["answer_relevance"].append(answer_relevance)
                results["context_relevance"].append(context_relevance)
                results["answer_correctness"].append(answer_correctness)
                
                print(f"Результаты для текущего вопроса:")
                print(f"faithfulness: {faithfulness}")
                print(f"answer_relevance: {answer_relevance}")
                print(f"context_relevance: {context_relevance}")
                print(f"answer_correctness: {answer_correctness}")
                
            except Exception as e:
                print(f"ОШИБКА при обработке вопроса: {str(e)}")
                print(f"Тип ошибки: {type(e)}")
                continue
            finally:
                thread.delete()
        
        # Вычисляем средние значения
        return {
            metric: np.mean(scores) if scores else 0.0 for metric, scores in results.items()
        }
    
    def _evaluate_metric(self, metric_type: str, *args) -> float:
        """Оценка конкретной метрики"""
        prompts = {
            "faithfulness": """
                Оцените, насколько ответ соответствует предоставленному контексту.
                Вопрос: {0}
                Контекст: {1}
                Ответ: {2}
                
                Оцените по шкале от 0 до 1, где:
                1 - ответ полностью основан на контексте
                0 - ответ не имеет отношения к контексту
                
                Верните только число.
            """,
            "answer_relevance": """
                Оцените, насколько ответ соответствует вопросу.
                Вопрос: {0}
                Ответ: {1}
                
                Оцените по шкале от 0 до 1, где:
                1 - ответ полностью отвечает на вопрос
                0 - ответ не имеет отношения к вопросу
                
                Верните только число.
            """,
            "context_relevance": """
                Оцените, насколько контекст релевантен вопросу.
                Вопрос: {0}
                Контекст: {1}
                
                Оцените по шкале от 0 до 1, где:
                1 - контекст полностью отвечает на вопрос
                0 - контекст не имеет отношения к вопросу
                
                Верните только число.
            """,
            "answer_correctness": """
                Оцените, насколько ответ соответствует эталонному ответу.
                Эталонный ответ: {0}
                Ответ: {1}
                
                Оцените по шкале от 0 до 1, где:
                1 - ответ полностью соответствует эталонному
                0 - ответ полностью не соответствует эталонному
                
                Верните только число.
            """
        }
        
        thread = self.sdk.threads.create(ttl_days=1, expiration_policy="static")
        try:
            prompt = prompts[metric_type].format(*args)
            thread.write(prompt)
            run = self.assistant.run(thread)
            response = run.wait()
            return float(response.text.strip())
        finally:
            thread.delete()

def main():
    # Загрузка тестовых данных
    test_data = []
    data_dir = Path(f'data_for_ragas_test/{lang}')
    
    print(f"Ищем файлы в директории: {data_dir.absolute()}")
    if not data_dir.exists():
        raise ValueError(f"Директория {data_dir} не существует")
    
    json_files = list(data_dir.glob("*.json"))
    print(f"Найдено JSON файлов: {len(json_files)}")
    
    for file in json_files:
        print(f"Обработка файла: {file}")
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"Добавлено {len(data)} Q&A пар из списка")
                    test_data.extend(data)
                elif isinstance(data, dict) and "qa_pairs" in data:
                    print(f"Добавлено {len(data['qa_pairs'])} Q&A пар из словаря")
                    test_data.extend(data["qa_pairs"])
                else:
                    print(f"Неизвестный формат данных в файле {file}")
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {str(e)}")
    
    if not test_data:
        raise ValueError("Не найдено тестовых данных в JSON файлах")
    
    print(f"Всего загружено {len(test_data)} тестовых примеров")
    
    # Инициализация оценщика
    folder_id = os.getenv("YANDEXGPT_FOLDER_ID")
    api_key = os.getenv("YANDEXGPT_API_KEY")
    
    if not folder_id or not api_key:
        raise ValueError(
            "Please set YANDEXGPT_FOLDER_ID and YANDEXGPT_API_KEY environment variables "
            "in .env file or system environment"
        )

    evaluator = YandexRAGEvaluator(folder_id=folder_id, api_key=api_key)
    
    # Оценка системы
    results = evaluator.evaluate_rag_system(test_data)
    
    # Вывод результатов
    print("\nРезультаты оценки RAG-системы:")
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")

if __name__ == "__main__":
    main() 