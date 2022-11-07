from ctypes import Array
from pyexpat import model
from tkinter import *
import tkinter as tk
import tkinter.filedialog as fd
import matplotlib
import matplotlib.pyplot as mp
import pandas as pd
import numpy as np
import openpyxl as op
import tensorflow
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping

root = Tk()
root.title("The first programm")
root.geometry("1050x500")


def Function1():
    filetypes = [('File', '.xlsx *.csv'), ('All files', '')]
    dlg = fd.Open(filetypes=filetypes)
    fl = dlg.show()
    if fl != '':
        message1_entry1.delete(0, END)
        message1_entry1.insert(0, fl)
    x, y = file_acceptance()

    model = Function_Model()
    Function_training(model, x, y)

    array = np.arange(45, 172, 1)

    model_prediction = model.predict(array)

    print(model_prediction)

    mp.plot(x, y, 'o', array, model.predict(array))
    mp.show()


def file_acceptance():
    url = message1_entry1.get()
    if (url.find(".xlsx", len(url) - 5) != -1):
        WS = pd.read_excel(url)
        WS_np = np.array(WS)
        print(WS.columns.ravel())
        num_rows, num_cols = WS_np.shape
        name_colum = WS.columns.ravel()
        x = np.ones((num_rows, num_cols - 1))
        y = WS_np[:, num_cols - 1]
        x = np.delete(WS_np, np.s_[-1:], axis=1)
        return x, y


message1 = StringVar()
message1_entry1 = Entry(textvariable=message1)
message1_entry1.place(relx=.6, rely=.5, anchor="c")

btn_file1 = Button(text="Plot", command=Function1)
btn_file1.place(relx=.6, rely=.1, anchor="c")


def Function_Model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(1280, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(640, activation='softmax'))
    model.add(BatchNormalization())
    model.add(Dense(1, 'elu'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    return model


def Function_training(Model_main, x, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=500)
    mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)
    history = Model_main.fit(x, y, epochs=3000, batch_size=40, shuffle=True, validation_split=0.0, validation_freq=2,
                             callbacks=[es, mc])
    return Model_main


# mp.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
# mp.show()
# x,y=metods.file_acceptance()

tk.mainloop()
