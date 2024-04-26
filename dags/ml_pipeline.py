from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pendulum


def prepare_dataset():
    import pandas as pd
    print("--- Inside prepare_dataset component ----")

    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv")
    df = df.dropna()
    df.to_csv(f'final_df.csv', index=False)


def train_test_split_dataset():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    print("--- Inside train_test_split_dataset component ----")
    final_df = pd.read_csv(f'final_df.csv')
    target_column = 'class'
    X = final_df.loc[:, final_df.columns != target_column]
    y = final_df.loc[:, final_df.columns == target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    np.save(f'X_train.npy', X_train)
    np.save(f'X_test.npy', X_test)
    np.save(f'y_train.npy', y_train)
    np.save(f'y_test.npy', y_test)


    print("\n--- X_train ---")
    print("\n")
    print(X_train)

    print("\n ---- X_test ----")
    print("\n")
    print(X_test)


    print("\n ---- y_train ----")
    print("\n")
    print(y_train)


    print("\n ---- y_test ----")
    print("\n")
    print(y_test)



def training_basic_model():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    print("----- inside training_basic_model component ----")

    X_train = np.load(f'X_train.npy', allow_pickle=True)
    y_train = np.load(f'y_train.npy', allow_pickle=True)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    import pickle
    with open(f'model_v1.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\n Logistic Regression Model is trained on diabetes dataset and saved to model.pkl ---- ")



def predict_on_test_data():
    import numpy as np
    import pandas as pd
    import pickle

    print("----- Inside predict_on_test_data component ----")
    with open(f'model_v1.pkl', 'rb') as f:
        model_logreg = pickle.load(f)
    
    X_test = np.load(f'X_test.npy', allow_pickle=True)
    y_pred = model_logreg.predict(X_test)
    np.save(f'y_pred.npy', y_pred)


    print("\n---- Predicted Classes ---")
    print("\n")
    print(y_pred)




def predict_prob_on_test_data():
    import numpy as np
    import pandas as pd
    import pickle

    print("----- Inside predict_on_test_data component ----")
    with open(f'model_v1.pkl', 'rb') as f:
        model_logreg = pickle.load(f)
    
    X_test = np.load(f'X_test.npy', allow_pickle=True)
    y_pred_prob = model_logreg.predict_proba(X_test)
    np.save(f'y_pred_prob.npy', y_pred_prob)


    print("\n---- Predicted Classes (Probability) ---")
    print("\n")
    print(y_pred_prob)



def get_metrics():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
    from sklearn import metrics
    print("---- Inside get_metrics component ----")
    y_test = np.load(f'y_test.npy', allow_pickle=True)
    y_pred = np.load(f'y_pred.npy', allow_pickle=True)
    y_pred_prob = np.load(f'y_pred_prob.npy', allow_pickle=True)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    entropy = log_loss(y_test, y_pred_prob)

    print(metrics.classification_report(y_test, y_pred))

    print("\n Model Metrics:", {"accuracy": round(acc, 2), "precision": round(prec, 2), "recall": round(recall, 2)})



with DAG(
    dag_id='ml_pipeline_airflow',
    schedule_interval='@hourly',
    start_date=datetime(2024, 4, 26),
    catchup=False
) as dag:
    
    task_prepare_datset = PythonOperator(
        task_id='prepare_dataset',
        python_callable = prepare_dataset,
    )

    task_train_test_split_dataset = PythonOperator(
        task_id='train_test_split_dataset',
        python_callable = train_test_split_dataset,
    )

    task_training_basic_model = PythonOperator(
        task_id='training_basic_model',
        python_callable = training_basic_model,
    )


    task_predict_on_test_data = PythonOperator(
        task_id='predict_on_test_data',
        python_callable = predict_on_test_data,
    )


    task_predict_prob_on_test_data = PythonOperator(
        task_id='predict_prob_on_test_data',
        python_callable = predict_prob_on_test_data,
    )


    task_get_metrics = PythonOperator(
        task_id='get_metrics',
        python_callable = get_metrics,
    )


    task_prepare_datset >> task_train_test_split_dataset >> task_training_basic_model >> task_predict_on_test_data >> task_predict_prob_on_test_data >> task_get_metrics
    
# if __name__ == "__main__":
#     prepare_dataset()
#     train_test_split_dataset()
#     training_basic_model()
#     predict_on_test_data()
#     predict_prob_on_test_data()
#     get_metrics()






