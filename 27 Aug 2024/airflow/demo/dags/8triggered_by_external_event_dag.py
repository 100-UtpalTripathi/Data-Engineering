from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="trigger_with_config_by_external_event",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    params={"file_name": "default/path"},
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def simple_function(params):
        print("params: ", params)

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={"params": "{{ params }}"},
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
