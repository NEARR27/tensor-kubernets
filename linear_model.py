import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import joblib  

print(tf.__version__)


margen, costo_base = 1.5, 20.0  
costo_produccion = np.linspace(10, 100, 200)
precio_venta = margen * costo_produccion + costo_base + np.random.normal(0, 2, 200)  


pares = list(zip(costo_produccion, precio_venta))
np.random.shuffle(pares)
costo_produccion, precio_venta = zip(*pares)
costo_produccion = np.array(costo_produccion)
precio_venta = np.array(precio_venta)

# Normalizar los datos
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(costo_produccion.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(precio_venta.reshape(-1, 1))


joblib.dump(scaler_x, 'scaler_x.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')


train_end = int(0.6 * len(X_scaled))
test_start = int(0.8 * len(X_scaled))
X_train, y_train = X_scaled[:train_end], y_scaled[:train_end]
X_test, y_test = X_scaled[test_start:], y_scaled[test_start:]
X_val, y_val = X_scaled[train_end:test_start], y_scaled[train_end:test_start]

tf.keras.backend.clear_session()


linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
linear_model.compile(optimizer=optimizer, loss='mean_squared_error')


history = linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)


test_points = np.array([30.0, 50.0, 70.0, 90.0]).reshape(-1, 1)
test_points_scaled = scaler_x.transform(test_points)
predictions_scaled = linear_model.predict(test_points_scaled).flatten()
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

print("Predicciones de precio de venta:", predictions)


export_path = 'linear-model/1/'
if not os.path.exists(export_path):
    os.makedirs(export_path)
tf.saved_model.save(linear_model, export_path)
