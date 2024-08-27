from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "raghul",
    "start_date": datetime(2024, 8, 27),
    "email": ["rvenkatachalam@presidio.com"],
    "retries": 3,
    "email_on_retry": False,
    "email_on_failure": False,
    "retry_delay": timedelta(seconds=5),
}


with DAG(
    dag_id="retry_example",
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def simple_function(retry):
        print("Retry count: ", retry)
        if int(retry) != 3:
            raise Exception("Raising dummy exception")

        print("I ran perfectly fine")

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={"retry": "{{ task_instance.try_number }}"},
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
