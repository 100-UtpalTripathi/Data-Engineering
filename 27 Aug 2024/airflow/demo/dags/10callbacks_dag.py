from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator


def task_failure_alert(context):
    print(f"Task has failed, task_instance_key_str: {context['task_instance_key_str']}")


def dag_success_alert(context):
    print(f"DAG has succeeded, run_id: {context['run_id']}")


default_args = {
    "owner": "raghul",
    "start_date": datetime(2024, 8, 27),
    "email": ["rvenkatachalam@presidio.com"],
    "retries": 1,
    "email_on_retry": False,
    "email_on_failure": False,
    "on_success_callback": dag_success_alert,
    "on_failure_callback": task_failure_alert,
    "retry_delay": timedelta(seconds=5),
}


with DAG(
    dag_id="callbacks_example",
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

        raise AirflowException("Raising dummy exception")

        # print("I ran perfectly fine")

    python_task = PythonOperator(
        task_id="python_operator",
        python_callable=simple_function,
        op_kwargs={"retry": "{{ task_instance.try_number }}"},
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    start_task >> python_task >> end_task
