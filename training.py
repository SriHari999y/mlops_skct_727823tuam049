# Name: Your Name | Roll No: 727823TUAM049
import mlflow
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_experiment("SKCT_727823TUAM049_FlightDelay")

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for i in range(12):
    with mlflow.start_run():
        seed = random.randint(1, 1000)
        start_time = time.time()

        model = RandomForestRegressor(
            n_estimators=100 + i*10,
            max_depth=5 + i,
            random_state=seed
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        training_time = time.time() - start_time

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("n_estimators", 100 + i*10)
        mlflow.log_param("max_depth", 5 + i)
        mlflow.log_param("random_seed", seed)
        mlflow.log_metric("training_time_seconds", training_time)

        mlflow.set_tags({
            "student_name": "Your Name",
            "roll_number": "727823TUAM049",
            "dataset": "FlightDelay"
        })

        mlflow.sklearn.log_model(model, "model")
