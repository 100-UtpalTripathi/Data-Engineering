from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="understanding_scheduling_behavior",
    schedule="25 7 * * *",
    catchup=True,
    start_date=datetime(2024, 8, 26),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def simple_function(**kwargs):
        for key, value in kwargs.items():
            print(f"{key}: {value}")

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={
            "data_interval_start": "{{ data_interval_start }}",
            "data_interval_end": "{{ data_interval_end }}",
            "execution_date": "{{ execution_date }}",
        },
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
