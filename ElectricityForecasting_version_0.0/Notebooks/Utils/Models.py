from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical computing
import glob  # Library for file handling
import re  # Library for regular expressions
from tqdm import tqdm  # Library for creating progress bars

import plotly.graph_objects as go  # Library for creating interactive plots

# Preprocessing
from sklearn import preprocessing  # Library for data preprocessing
import keras  # Deep learning library
from keras.models import Sequential  # Sequential model for stacking layers
from keras.layers.core import Dense, Dropout, Activation  # Layers for fully connected neural networks
from keras.layers import LSTM  # LSTM layer for sequence modeling
from keras.models import load_model  # Loading pre-trained models
import matplotlib.pyplot as plt  # Library for basic data visualization
import h5py  # Library for handling large datasets
import datetime  # Library for date and time operations
import tensorflow as tf  # Deep learning library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from joblib import dump, load
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from datetime import datetime
import os

def bi_directional_lstm_model(param,train_input,train_output,val_input,val_output):
   
   configFileName ='../ConfigFiles' 
   os.makedirs(f'{configFileName}',exist_ok=True)

   model = Sequential()
   model.add(Bidirectional(LSTM(param['lstm_units'], return_sequences=True), input_shape=param['input_shape']))
   model.add(Bidirectional(LSTM(param['lstm_units'], return_sequences=True), ))

   model.add(Flatten())
   model.add(Dense(param['dense_units'], activation='relu'))
   model.add(Dense(param['num_forecast_hours'], activation='linear'))

   # Compile the model
   optimizer = Adam(learning_rate=param['learning_rate'])
   model.compile(optimizer=optimizer, loss='mean_squared_error')

   # print(model.summary())
   
   
   
   optimizer = tf.keras.optimizers.Adam(learning_rate=param['learning_rate'])
   # Compile the model
   model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=optimizer)
   
   # Get the model's parameters
   model_params = model.count_params()

   # Create the model name based on parameters and training settings
   model_name = f"model_{param['epochs']}_epochs_{param['batch_size']}_{model_params}_params_input_window_{param['num_input_days']}"

   # Define early stopping callback
   early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

   # Define TensorBoard callback with model-specific log folder
   tb_callback = TensorBoard(log_dir=f'logs/{model_name}/', write_graph=False, update_freq='epoch')

   # Define model checkpoint callback with model-specific filename
   checkpoint_callback = ModelCheckpoint(f'checkpoints/{model_name}_{{epoch:02d}}.h5', save_weights_only=True, save_best_only=True)

   # Train the model with early stopping, checkpoints, and TensorBoard
   history = model.fit(train_input, train_output, #validation_data=(x_val, y_val), 
                     epochs=param['epochs'], 
                     batch_size=param['batch_size'], 
                     verbose=2,
                     validation_data=(val_input, val_output),
                     callbacks=[early_stopping, 
                              tb_callback, 
                              checkpoint_callback])

   current_time = datetime.now()

   # Format the current time
   formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
   print('Saveing Model Successfully')
   dump(model, f'{configFileName}/{formatted_time}_bidirectionalLstm.pkl')


   return model