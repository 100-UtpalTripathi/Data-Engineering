from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="simple_python_operator",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def simple_function(run_id):
        print("Run ID: ", run_id)
        print("{{ run_id }}")

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={"run_id": "{{ run_id }}"},
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
