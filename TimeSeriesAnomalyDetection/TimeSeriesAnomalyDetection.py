import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

path = '../data/ambient_temperature_system_failure.csv'
data = pd.read_csv(path)

data_values = data.drop('timestamp', axis=1).values
data_values = data_values.astype('float32')
data_converted = pd.DataFrame(data_values, columns=data.columns[1:])
data_converted.insert(0, 'timestamp', data['timestamp'])

path = '../data/ambient_temperature_system_failure.csv'
data = pd.read_csv(path)


data_values = data.drop('timestamp', axis=1).values
data_values = data_values.astype('float32')
data_converted = pd.DataFrame(data_values, columns=data.columns[1:])
data_converted.insert(0, 'timestamp', data['timestamp'])

data_tensor = tf.convert_to_tensor(data_converted.drop('timestamp', axis=1).values, dtype=tf.float32)
input_dim = data_converted.shape[1] - 1
encoding_dim = 10

# autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='relu')(encoder)
autoencoder = Model(inputs = input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_tensor, data_tensor, epochs=50, batch_size=32, shuffle=True)

# reconstruction error
reconstructions = autoencoder.predict(data_tensor)
mse = tf.reduce_mean(tf.square(data_tensor - reconstructions), axis=1)

anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores')
anomaly_scores.index = data_converted.index

# anomaly detection threshold
threshold = anomaly_scores.quantile(0.99)
anomalous = anomaly_scores > threshold
binary_labels = anomalous.astype(int)
precision, recall, f1_score, support = precision_recall_fscore_support(binary_labels, anomalous, average='binary')

test = data_converted['value'].values
predictions = anomaly_scores.values

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
print("Support: ", support)

plt.figure(figsize=(16, 8))
plt.plot(data_converted['timestamp'],
         data_converted['value'])
plt.plot(data_converted['timestamp'][anomalous],
         data_converted['value'][anomalous], 'ro')
plt.title('Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()