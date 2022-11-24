import clf as clf
import joblib
from sklearn.ensemble import IsolationForest
import PySimpleGUI as sg
import pandas as pd
# import joblib from sklearn.externals
import csv
# Define the window's contents
layout = [[sg.Text("Please enter the duration_: ")],
        [sg.Input(key='-INPUT1-')],
        [sg.Text("Please enter the src_bytes: ")],
        [sg.Input(key='-INPUT2-')],
        [sg.Text("Please enter the dst_bytes: ")],
        [sg.Input(key='-INPUT3-')],
        [sg.Text(size=(40,1), key='-OUTPUT-')],
        [sg.Button('Predict'), sg.Button('Quit')]]

# Create the window
window = sg.Window('Anomaly Detection', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    if int(values['-INPUT1-']) > 256670 or int(values['-INPUT1-'])<0:
        window['-OUTPUT-'].update("Error: Please enter a number in the range [0-256670]", background_color='white',
                                  text_color='red')
    df = pd.read_csv("conn_attack.csv", names=[ "duration_", "src_bytes", "dst_bytes"], header=None)
    df
    new_row = {'duration_': values['-INPUT1-'], 'src_bytes': values['-INPUT2-'], 'dst_bytes': values['-INPUT3-']}

    # append row to the dataframe
    df= df.append(new_row, ignore_index=True)
    input_attributes = df[["duration_", "src_bytes", "dst_bytes"]]
    isolation_forest = IsolationForest(contamination=0.05, n_estimators=100)
    isolation_forest.fit(input_attributes.values)
    df["anomaly"] = pd.Series(isolation_forest.predict(input_attributes.values))
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})
    # print(df["anomaly"].value_counts())
    print(df.tail())
    if df['anomaly'].iloc[-1]== 1:
            window['-OUTPUT-'].update("Anomalous point", background_color='white',
                              text_color='red')
    elif df['anomaly'].iloc[-1] == 0:
                window['-OUTPUT-'].update("Not an anomalous point", background_color='white',
                              text_color='green')

    # file.close()



naiveBayesModel = open("models/naivemodel.pkl", "wb")
joblib.dump(df,naiveBayesModel)
naiveBayesModel.close()
# Finish up by removing from the screen
window.close()

