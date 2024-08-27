from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="variables_example",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def simple_function(**kwargs):
        data_training = Variable.get("data_training")
        print("From Variable class -> data_training: ", data_training)
        print("Through jinja injection ->  data_training: ", kwargs["data_training"])

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={"data_training": "{{ var.value.data_training }}"},
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
