import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = load_wine()
X = df.data
y = df.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#params
max_depth = 10
n_estimators = 10

with mlflow.start_run():
    model = RandomForestClassifier(max_depth = max_depth, n_estimators=n_estimators)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test,y_pred),
               "precision": precision_score(y_test,y_pred,average="weighted"),
               "recall": recall_score(y_test,y_pred,average="weighted"),
               "f1_score": f1_score(y_test,y_pred,average="weighted")
    }
    print(metrics)
    mlflow.log_metrics(metrics)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimators",n_estimators)