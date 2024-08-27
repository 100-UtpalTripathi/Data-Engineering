from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

with DAG(
    dag_id="branch_operator_example",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def save_to_s3():
        import random

        choice = random.choice(["s3://some/s3_uri", None])
        return choice

    save_to_s3_task = PythonOperator(task_id="save_to_s3", python_callable=save_to_s3)

    def branch_function(s3_uri):
        if s3_uri is not None:
            return "process_s3"
        return "end"

    branch_task = BranchPythonOperator(
        task_id="branch",
        python_callable=branch_function,
        op_kwargs={"s3_uri": save_to_s3_task.output},
    )

    def process_s3(**kwargs):
        task_instance = kwargs["ti"]
        s3_uri = task_instance.xcom_pull(task_ids="save_to_s3")
        print("Processing s3_uri: ", s3_uri)

    process_s3_task = PythonOperator(
        task_id="process_s3", python_callable=process_s3, provide_context=True
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> save_to_s3_task >> branch_task >> [process_s3_task, end_task]
