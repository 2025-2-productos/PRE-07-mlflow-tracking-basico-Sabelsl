import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)

y = df["quality"]
X = df.drop(columns=["quality"])

logged_model = "runs:/23130ed670f648a0990d32f5b0c77507/model"
logged_model = mlflow.pyfunc.load_model(logged_model)
predictions = logged_model.predict(X)
print(predictions)
