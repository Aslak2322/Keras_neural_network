import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "admissions_data.csv")

df = pd.read_csv(file_path)

print(df.columns)
print(df.head())

features = df.drop('Chance of Admit ', axis=1)
labels = df['Chance of Admit ']
print(labels[0:5])

df.info()

X_train, X_test, y_train, y_test = train_test_split(
  features, labels,
  random_state=1,
  test_size=0.2,
)

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

def build_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1))
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

my_model = build_model()

history = my_model.fit(
  X_train,
  y_train,
  batch_size=25,
  epochs=35,
  validation_split=0.2 
)
loss, mae = my_model.evaluate(X_test, y_test)

# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

# used to keep plots from overlapping each other  
fig.tight_layout()
plt.show()

# note these variable names may differ from yours
predicted_values = my_model.predict(x_test) 
print(r2_score(y_test, predicted_values)) 
