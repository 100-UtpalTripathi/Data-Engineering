from datetime import datetime

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

with DAG(
    dag_id="parallel_tasks_example",
    schedule=None,
    catchup=False,
    start_date=datetime(2024, 8, 27),
    tags=["basic"],
) as dag:
    start_task = EmptyOperator(
        task_id="start",
    )

    def task_one():
        import time

        time.sleep(2)

    task_one_task = PythonOperator(task_id="task_one", python_callable=task_one)

    def task_two():
        import time

        time.sleep(4)

    task_two_task = PythonOperator(task_id="task_two", python_callable=task_two)

    def task_three():
        import time

        time.sleep(6)

    task_three_task = PythonOperator(task_id="task_three", python_callable=task_three)

    def task_four():
        import time

        time.sleep(8)
        raise AirflowFailException()

    task_four_task = PythonOperator(task_id="task_four", python_callable=task_four)

    end_task = EmptyOperator(task_id="end", 
                             trigger_rule=TriggerRule.ALL_DONE
                             )

    (
        start_task
        >> [task_one_task, task_two_task, task_three_task, task_four_task]
        >> end_task
    )
