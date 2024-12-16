# <YOUR_IMPORTS>
import os
import dill
import pandas as pd
import json
import platform

# Универсальный путь в зависимости от ОС
if platform.system() == "Windows":
    path = os.environ.get("PROJECT_PATH", ".")
else:  # Для Linux или других систем
    path = os.environ.get("PROJECT_PATH", "/opt/airflow/dags/plugins")


def predict():
    # <YOUR_CODE>
    # Определение путей к файлам
    models_dir = f'{path}/data/models'
    test_data_dir = f'{path}/data/test'
    last_model_file = f'{path}/data/last_model.txt'

    # Проверка наличия файла last_model.txt и чтение его содержимого
    if os.path.exists(last_model_file):
        with open(last_model_file, 'r') as file:
            model_name = file.read().strip()

        # Путь модели
        model_path = f'{models_dir}/{model_name}'
        predictions_path = f'{path}/data/predictions/{model_name.split('.')[0]}_predicts.csv'

        # Загрузка модели
        with open(model_path, 'rb') as file:
            model = dill.load(file)

        # Обнаружение всех JSON-файлов в директории
        json_files = [f for f in os.listdir(test_data_dir) if f.endswith('.json')]

        # создадим словарь предсказаний
        predictions = dict()
        # Загрузка и объединение тестовых данных

        for json_file in json_files:
            file_path = f'{test_data_dir}/{json_file}'

            # Читаем JSON-файл
            with open(file_path, 'r') as file:
                json_data = json.load(file)

            # Создаем DataFrame из объекта JSON
            test_data = pd.DataFrame.from_dict([json_data])
            # Предсказание
            f = lambda x: 'low' if x == 0.0 else 'medium' if x == 1.0 else 'high'
            predictions[json_file] = f(model['model'].predict(test_data)[0])

        # Создание DataFrame с предсказаниями
        predictions_df = pd.DataFrame(list(predictions.items()), columns=['files', 'predictions'])

        # Сохранение предсказаний в CSV
        predictions_df.to_csv(predictions_path, index=False)
    else:
        exit(1)  # Остановка программы с кодом ошибки 1


if __name__ == '__main__':
    predict()
