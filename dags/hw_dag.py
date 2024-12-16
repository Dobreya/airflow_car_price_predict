import os
import sys
import datetime as dt

from airflow import DAG  # Импорт DAG
from airflow.operators.python_operator import PythonOperator


path = os.path.expanduser('/opt/airflow/dags/plugins')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict
# <YOUR_IMPORTS>

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2024, 12, 16),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="15 * * * *",
        default_args=args,
) as dag:
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )
    # <YOUR_CODE>

