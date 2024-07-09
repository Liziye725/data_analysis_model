import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

path = './data/ambient_temperature_system_failure.csv'
data = pd.read_csv(path)

data_values = data.drop('timestamp', axis=1).values

data_values = data_values.astype('float32')
data_converted = pd.DataFrame(data_values, columns=data.columns[1:])

data_converted.insert(0, 'timestamp', data['timestamp'])