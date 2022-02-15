import scipy
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
import math


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from keras.models import Sequential
from keras.layers import Dense


from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(1234) 

# Read file
file = 'p1p2s1.csv'
raw_data = pd.read_csv(file)
df = raw_data.copy() 
df.head()
df.shape
df.describe()

# Define a function to draw time_series plot
def timeseries (x_axis, y_axis, x_label, y_label):
    plt.figure(figsize = (10, 6))
    plt.plot(x_axis, y_axis, color ='black')
    plt.xlabel(x_label, {'fontsize': 12}) 
    plt.ylabel(y_label, {'fontsize': 12})
timeseries(df.index, df['RPE'] ,'Time (Seconds)','RPE (Ratings of Perceived Exertions)')

# Check missing values
df.isnull().sum() 

# Replace missing values by interpolation
def replace_missing (attribute):
    return attribute.interpolate(inplace=True)
replace_missing(df['time'])
replace_missing(df['sensor1_acc'])
replace_missing(df['sensor1_jerk'])
replace_missing(df['RPE']) 

# Split train data and test data
train_size = int(len(df)*0.8)
train_dataset, test_dataset = df.iloc[:train_size],df.iloc[train_size:]


# Plot train and test data
plt.figure(figsize = (10, 6))
plt.plot(train_dataset.RPE)
plt.plot(test_dataset.RPE)
plt.xlabel('Time (Seconds)')
plt.ylabel('RPE (Ratings of Perceived Exertions)')
plt.legend(['Train set', 'Test set'], loc='upper right')
print('Dimension of train data: ',train_dataset.shape)
print('Dimension of test data: ', test_dataset.shape) 


# Split train data to X and y
X_train = train_dataset.drop('RPE', axis = 1)
y_train = train_dataset.loc[:,['RPE']]
# Split test data to X and y
X_test = test_dataset.drop('RPE', axis = 1)
y_test = test_dataset.loc[:,['RPE']]  



# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))

# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)

# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)

# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)


# Create a 3D input
def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)
TIME_STEPS = 30
X_test, y_test = create_dataset(test_x_norm, test_y_norm,   
                                TIME_STEPS)
X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                  TIME_STEPS)
print('X_train.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_train.shape)



# Create LSTM or GRU model
def create_model(units, m):
    model = Sequential()
    model.add(m (units = units, return_sequences = True,
                input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2))
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model 



# LSTM

model_lstm = create_model(64, LSTM)

# Fit BiLSTM, LSTM and GRU
def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)
    history = model.fit(X_train, y_train, epochs = 25,  
                        validation_split = 0.2, batch_size = 64, 
                        shuffle = False, callbacks = [early_stop])
    return history


history_lstm = fit_model(model_lstm)


# evaluate the model
train_acc = model_lstm.evaluate(X_train, y_train, verbose=0) 
print('train_acc: %f' % train_acc)


test_acc = model_lstm.evaluate(X_test, y_test, verbose=0)
print('test_acc: %f' % test_acc)


# predict probabilities for test set
yhat_probs = model_lstm.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model_lstm.predict(X_test>0.5).astype("int32")


# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]




# Plot train loss and validation loss
def plot_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')


plot_loss (history_lstm)


y_test = scaler_y.inverse_transform(y_test)
y_train = scaler_y.inverse_transform(y_train) 

# Make prediction
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction


prediction_lstm = prediction(model_lstm)


#zoom
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Plot true future vs prediction
def plot_future(prediction, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='True Future')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('RPE (Ratings of Perceived Exertions)')

plot_future(prediction_lstm, y_test)


# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print(model + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')

evaluate_prediction(prediction_lstm, y_test, 'LSTM')




























