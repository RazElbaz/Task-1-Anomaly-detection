import clf as clf
import joblib
from sklearn.ensemble import IsolationForest
import PySimpleGUI as sg
import pandas as pd
# import joblib from sklearn.externals
import csv
# Define the window's contents


df = pd.read_csv("conn_attack.csv", names=["duration_", "src_bytes", "dst_bytes"], header=None)
df.size
df.dtypes
df.isnull().isnull().sum()


def predict(a,b,c):
    df = pd.read_csv("conn_attack.csv", names=["duration_", "src_bytes", "dst_bytes"], header=None)
    new_row = {'duration_': a, 'src_bytes': b, 'dst_bytes':c}
    # append row to the dataframe
    df = df.append(new_row, ignore_index=True)
    input_attributes = df[["duration_", "src_bytes", "dst_bytes"]]
    isolation_forest = IsolationForest(contamination=0.05, n_estimators=100)
    isolation_forest.fit(input_attributes.values)
    df["anomaly"] = pd.Series(isolation_forest.predict(input_attributes.values))
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})
    # print(df["anomaly"].value_counts())
    print(df.tail())
    if df['anomaly'].iloc[-1]== 1:
        print("Anomalous point")
        return 1
    elif df['anomaly'].iloc[-1] == 0:
        return 0
        print("Not an anomalous point")

    # file.close()



naiveBayesModel = open("models/naivemodel.pkl", "wb")
joblib.dump(df,naiveBayesModel)
naiveBayesModel.close()
    # Finish up by removing from the screen


print(df.tail())
