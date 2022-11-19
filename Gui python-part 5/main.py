import PySimpleGUI as sg
import pandas as pd
import csv
# Define the window's contents
layout = [[sg.Text("Please enter the record ID for which you would like to make a prediction")],
          [sg.Input(key='-INPUT-')],
          [sg.Text(size=(40,1), key='-OUTPUT-')],
          [sg.Button('Ok'), sg.Button('Quit')]]

# Create the window
window = sg.Window('Anomaly Detection', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    if int(values['-INPUT-']) > 256670 or int(values['-INPUT-'])<0:
        window['-OUTPUT-'].update("Error: Please enter a number in the range [0-256670]", background_color='white',
                                  text_color='red')
    file = open("results.csv", "r")
    read_file = csv.reader(file)

    # loop through the csv list
    for s in read_file:
        if s[0] == values['-INPUT-'] :
            if s[1]== str(1):
                window['-OUTPUT-'].update("Anomalous point", background_color='white',
                              text_color='red')
            elif s[1] == str(0):
                window['-OUTPUT-'].update("Not an anomalous point", background_color='white',
                              text_color='green')

    file.close()


# Finish up by removing from the screen
window.close()