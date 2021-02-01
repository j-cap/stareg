
# This is a test GUI

import PySimpleGUI as sg
import pandas as pd
import numpy as np
from matplotlib import use as use_agg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

from stareg import Stareg

FONT = "Helvetica 14"
CONSTR = ('inc', 'dec', 'peak', 'valley', 'conc', 'conv', 'none')

# Use Tkinter Agg
use_agg('TkAgg')

def load_data():
    # Button to browse for the data file
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Dataset (a CSV file)', size=(16, 1)),sg.InputText(),
               sg.FileBrowse(file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))],
               [sg.Submit(), sg.Cancel()]]
    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
    except Exception:
        pass
    fname = values[0]
    try: 
        df = pd.read_csv(fname, sep=",", engine="python")
        headers = list(df.columns)
        window1.close()
        return (df, headers, fname.split("/")[-1])
    except Exception:
        sg.popup("Error reading file, try again")
        # window1.close()

def fit_data():
    # Button to get the parameter values for the model and fit it
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text("Enter the number of basis functions: ", size=(30,1)), 
               sg.InputText("100", key="-NR_SPLINES-")],
              [sg.Text("Enter the constraint: ",  size=(30,1)), sg.Combo([*CONSTR], default_value="none", key="-CONSTR_CHOSEN-")],
              [sg.Text("Enter the knot placement: ",  size=(30,1)), 
               sg.Combo(['equidistant', 'quantile'], default_value="equidistant", key="-KNOTS_CHOSEN-")],
             [sg.Submit("Ok")],
    ]
    window_fit = sg.Window("Set Options", layout)
    try:
        event, values = window_fit.read()
    except Exception:
        pass
    print(values)
    nr_splines = int(values["-NR_SPLINES-"])
    constraint =  values["-CONSTR_CHOSEN-"]
    knot_type = values["-KNOTS_CHOSEN-"][0]
    try:
        print((nr_splines, constraint, knot_type))
        window_fit.close()
        return (nr_splines, constraint, knot_type)
    except Exception:
        sg.popup("Error saving parameters, try again.")
        print((nr_splines, constraint, knot_type))

def save_model(model):
    # Button to save the trained model
    sg.set_options(auto_size_buttons=True)
    layout = [[
        sg.InputText(key='-FILE_TO_SAVE-', default_text='**.pkl', enable_events=True),
        sg.InputText(key="-SAVE_AS-", do_not_clear=False, enable_events=True, visible=False),
        sg.FileSaveAs(initial_folder='/tmp')
    ]]
    window_save = sg.Window("Save", layout)

    while True:
        event, values = window_save.Read()
        print("event: ", event, "values: ", values)
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        elif event == "-SAVE_AS-":
            fname = values["-SAVE_AS-"]
            if fname:
                window_save["-FILE_TO_SAVE-"].update(value=fname)
                pd.to_pickle(model, fname)
                window_save.close()

    
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack() #side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig):
    fig.get_tk_widget().forget()
    plt.close('all')

layout = [
    [sg.Text("Welcome to STAREG", size=(25,1), key="-HELLO-", font=FONT)], #     sg.Button('Plot', key="-plot-", size=(20,2)), sg.Button('Exit')], 
    [sg.Button("Load data", size=(20,2), key="-load-", enable_events=True, font=FONT), 
     sg.Button("Fit", size=(20,2), key="-fit-", enable_events=True, font=FONT),
     sg.Button("Save Model", size=(20,2), key="-save-", enable_events=True, font=FONT)],
    [sg.Text("", size=(50,1),key='-loaded-', pad=(5,5), font='Helvetica 14')],
    [sg.T('Figure:')],
    [sg.Canvas(key='-CANVAS-')],
    [sg.B('Ok')]
]

window = sg.Window("Test GUI", layout, size=(900,800), finalize=True)# , element_justification='center')

# Default settings for matplotlib graphics
fig, ax = plt.subplots()
fig_agg = None
model = None
df = None

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break

    # event to load the data from a .csv file
    if event == "-load-":
        df, header, fname = load_data()
        print(df.head())
        if type(df) == pd.DataFrame:
            window["-loaded-"].update(f"Dataset loaded: {fname}")
        
        time.sleep(0.1)  # Plot the loaded dataset
        if 'fig_canvas_agg' in locals():
            delete_fig_agg(fig_canvas_agg)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))           
        ax.scatter(df["x"],df["y"], c="r", marker="x")
        ax.scatter(df["x"],df["ytrue"], c="k", marker="1")
        ax.legend(["Noisy Data", "True Data"])
        ax.set_title(f"Dataset {fname}")
        ax.grid()
        # add the plot to the window
        fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    # event to get the parameters and fit the model
    if event == "-fit-":
        if df is None:
            sg.popup("No data available, please run 'Load Data' first!")
            continue
        nr_splines, constraint, knot_type = fit_data()
        time.sleep(0.2)
        descr = ( ("s(1)", nr_splines, constraint, 6000, knot_type), )
        X, y = df["x"].values.reshape((-1,1)), df["y"].values     # get data
        print(f"Model Description: \n {descr}".center(20, "-"))   # fit model
        STAREG = Stareg()
        model = STAREG.fit(description=descr, X=X, y=y)
        Xpred = np.linspace(X.min(), X.max(), 100)
        ypred = STAREG.predict(Xpred.reshape(-1,1), model["model"], model["coef_"])
        if 'fig_canvas_agg' in locals():
            delete_fig_agg(fig_canvas_agg)          # plot fit
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16,9), gridspec_kw={"height_ratios":[0.8,0.2]})
        ax1, ax2 = axs
        ax1.scatter(df["x"], df["y"], c="k", marker="x", label="Data")
        ax1.scatter(df["x"], df["ytrue"], c="red", marker="1", label="True Function")
        ax1.plot(Xpred, ypred, c="blue", label="Fit")
        ax1.legend()
        ax1.grid()
        ax2.scatter(df["x"], df["y"] - model["B"]@model["coef_"], c="red", marker="1")
        ax2.legend(["Residual"])
        ax2.grid()
        fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    # event to save the model after it is fitted
    if event == "-save-":
        if model is not None:
            save_model(model=model)
        else:
            sg.popup("No model available, please run 'Fit' first!")


window.close()