from datetime import datetime

from airflow import DAG, XComArg
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="xcom_example",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def explicit_push(**kwargs):
        task_instance = kwargs["ti"]
        task_instance.xcom_push(key="explicit_push", value="explicit_value")

    explicit_xcom_push_task = PythonOperator(
        task_id="explicit_xcom_push",
        python_callable=explicit_push,
        provide_context=True,
    )

    def implicit_push():
        return {"implicit_push": "implicit_value"}

    implicit_xcom_push_task = PythonOperator(
        task_id="implicit_xcom_push", python_callable=implicit_push
    )

    def xcom_pull(**kwargs):
        through_xcom_arg = kwargs["through_xcom_arg"]
        task_instance = kwargs["ti"]
        explicit_push_value = task_instance.xcom_pull(
            key="explicit_push", task_ids="explicit_xcom_push"
        )
        implicit_push_value = task_instance.xcom_pull(task_ids="implicit_xcom_push")
        print("through_xcom_arg: ", through_xcom_arg)
        print("explicit_push_value: ", explicit_push_value)
        print("implicit_push_value: ", implicit_push_value)

    xcom_pull_task = PythonOperator(
        task_id="xcom_pull",
        python_callable=xcom_pull,
        op_kwargs={
            "through_xcom_arg": implicit_xcom_push_task.output,
        },
        provide_context=True,
    )

    end_task = EmptyOperator(
        task_id="end",
    )

    (
        start_task
        >> explicit_xcom_push_task
        >> implicit_xcom_push_task
        >> xcom_pull_task
        >> end_task
    )
