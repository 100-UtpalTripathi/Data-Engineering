from airflow.models import DAG
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
import panda as pd

import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
import numpy as np

def prepare_data():
    import pandas as pd

    df = pd.read_csv('https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv')
    df = df.dropna()

    df.to_csv(f'final_df.csv', index=False)


def train_test_split():
    import pandas as pd

    df = pd.read_csv('final_df.csv')
    target_column = 'class'
    X = df.loc[:, df.columns != target_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv(f'X_train.csv', index=False)
    X_test.to_csv(f'X_test.csv', index=False)
    y_train.to_csv(f'y_train.csv', index=False)
    y_test.to_csv(f'y_test.csv', index=False)

def training_basic_classifier():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np

    X_train = np.load(f'X_train.npy', allow_pickle=True)
    y_train = np.load(f'y_train.npy', allow_pickle=True)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    import pickle

    # y_train_pred = model.predict(X_train)
    # accuracy = accuracy_score(y_train, y_train_pred)

    with open(f'model.pkl', 'wb') as f:
        pickle.dump(model, f)


def predict_on_test_data():
    import pandas as pd
    import pickle
    import numpy as np

    with open(f'model.pkl', 'rb') as f:
        model = pickle.load(f)

    X_test = np.load(f'X_test.npy', allow_pickle=True)
    y_pred = model.predict(X_test)
    np.save(f'y_pred.npy', y_pred)

with DAG(
    dag_id='ml_pipeline',
    schedule_interval='@daily',
    start_date=datetime(2024, 8, 27),
    catchup=False ) as dag:

    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data
    )

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split
    )

    task_training_basic_classifier = PythonOperator(
        task_id='training_basic_classifier',
        python_callable=training_basic_classifier
    )

    task_predict_on_test_data = PythonOperator(
        task_id='predict_on_test_data',
        python_callable=predict_on_test_data
    )

    task_prepare_data >> task_train_test_split >> task_training_basic_classifier >> task_predict_on_test_data
    