from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_housing import download_data, preprocess_data, train_model

default_args = {
    'owner': 'me',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'housing_price_pipeline',
    default_args=default_args,
    description='Полный pipeline для прогноза цен на жилье',
    schedule_interval=timedelta(days=1),
)

download_task = PythonOperator(
    task_id='download_housing_data',
    python_callable=download_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_housing_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_housing_model',
    python_callable=train_model,
    dag=dag,
)

download_task >> preprocess_task >> train_task
