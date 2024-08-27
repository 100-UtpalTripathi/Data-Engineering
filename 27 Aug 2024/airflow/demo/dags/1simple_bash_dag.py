from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="simple_bash_operator_dag",
    schedule=None,
    start_date=datetime(2024, 8, 27),
    catchup=False,
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(task_id="start")

    bash_task = BashOperator(
        task_id="bash_operator",
        bash_command="echo 'Run ID: {{ run_id }} from BashOperator'",
    )

    end_task = EmptyOperator(task_id="end")

    start_task >> bash_task >> end_task
