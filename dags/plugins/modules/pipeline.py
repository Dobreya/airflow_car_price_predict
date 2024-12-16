import logging
import os
from datetime import datetime
import platform

import dill
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Универсальный путь в зависимости от ОС
if platform.system() == "Windows":
    path = os.environ.get("PROJECT_PATH", ".")
else:  # Для Linux или других систем
    path = os.environ.get("PROJECT_PATH", "/opt/airflow/dags/plugins")


def pipeline():
    def best_pipe():
        # функция удаления ненужных колонок
        def filter_data(df):
            df_copy = df.copy(deep=True)
            columns_to_drop = [
                'id',
                'url',
                'region',
                'region_url',
                'price',
                'manufacturer',
                'image_url',
                'description',
                'posting_date',
                'lat',
                'long'
            ]

            return df_copy.drop(columns_to_drop, axis=1)

        # удаление выбросов
        def calculate_outliers(data):
            q25 = data.quantile(0.25)
            q75 = data.quantile(0.75)
            iqr = q75 - q25
            boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            return boundaries

        def outliers_transform(df):
            df_copy = df.copy(deep=True)
            boundaries = calculate_outliers(df_copy['year'])  # считаем границы
            df_copy.loc[df_copy['year'] < boundaries[0], 'year'] = round(
                boundaries[0])  # заменяем выбросы нижней границей
            df_copy.loc[df_copy['year'] > boundaries[1], 'year'] = round(
                boundaries[1])  # заменяем выбросы верхней границей
            return df_copy

        # Создание новых предикторов

        def create_features(df):
            df_copy = df.copy(deep=True)
            # Добавляем фичу "short_model" – это первое слово из колонки model
            df_copy.loc[:, 'short_model'] = df_copy['model'].apply(lambda x: str(x).lower().split(' ')[0] if x else x)

            # Добавляем фичу "age_category" (категория возраста)
            df_copy.loc[:, 'age_category'] = df_copy['year'].apply(
                lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
            return df_copy

        logging.info('Car Price Category Prediction Pipeline')
        # открываем наши данные, разделяем целевую переменную - у и обрабатываемые
        df = pd.read_csv(f'{path}/data/train/homework.csv')

        X = df.drop('price_category', axis=1)
        price_cat_dict = {'low': 0, 'medium': 1, 'high': 2}
        y = df['price_category'].map(price_cat_dict)

        functional_transformer = Pipeline(steps=[
            # Удаление ненужных колонок
            ('filter', FunctionTransformer(filter_data)),
            # удаление выбросов в колонке year.
            ('outliers', FunctionTransformer(outliers_transform)),
            # Создание 2-х новых предикторов
            ('create_features', FunctionTransformer(create_features)),
        ])

        numerical_features = functional_transformer.fit_transform(X).select_dtypes(include=['int64', 'float64']).columns
        categorical_features = functional_transformer.fit_transform(X).select_dtypes(include=['object']).columns

        numerical_transformer = Pipeline(steps=[
            # заполнение пропусков медианным значением
            ('imputer', SimpleImputer(strategy='median')),
            # масштабирование количественных предикторов
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            # заполнение пропусков категориальных переменных
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # кодирование категориальных переменных
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        # Активируем модели (Регрессия, Случайный лес, Опорные вектора
        models = (
            LogisticRegression(solver='liblinear'),
            RandomForestClassifier(),
            SVC()
        )

        best_score = .0
        best_pipe = None

        for model in models:
            pipe = Pipeline(steps=[
                ('functional_transformer', functional_transformer),
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
            logging.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

            if score.mean() > best_score:
                best_score = score.mean()
                best_pipe = pipe

        logging.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

        # Обучим лучшую модель на всём датасете
        best_pipe.fit(X, y)

        return best_pipe, type(best_pipe.named_steps['classifier']).__name__, best_score

    # создаём модель`
    best_pipe = best_pipe()

    # сериализуем модель с помощью dill:

    # путь и имя модели, где будет сохранена модель
    model_name = datetime.now().strftime("%Y%m%d%H%M")
    model_filename = f'{path}/data/models/cars_pipe_{model_name}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': best_pipe[0],

            'metadata': {
                'name': 'car_price_category_prediction',
                'author': 'Dobreja',
                'version': '999.1',
                'date': datetime.now(),
                'model_type': best_pipe[1],
                'accuracy': best_pipe[2]
            }
        }, file
        )
    logging.info(f'Model is saved as {model_filename}')

    # Путь к файлу last_model.txt
    last_model_file_path = f'{path}/data/last_model.txt'

    # Записываем строку в файл last_model.txt
    with open(last_model_file_path, 'w') as file:
        file.write(f'cars_pipe_{model_name}.pkl')