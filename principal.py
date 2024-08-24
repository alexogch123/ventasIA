import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

# Ruta del archivo
file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\historial ventas.csv'

# Leer el archivo CSV con codificación UTF-8
df = pd.read_csv(file_path, encoding='utf-8')

# Renombrar columnas y combinar para crear la columna de fecha
df.rename(columns={'Month': 'MES', 'CVE PRODUCTO': 'PRODUCTO', 'LBS': 'cantidad_vendida'}, inplace=True)
df['FECHA'] = pd.to_datetime(df['AÑO'].astype(str) + '-' + df['MES'].astype(str) + '-01')

# Filtrar datos
df_grouped = df.groupby(['PRODUCTO', 'DESCRIPCION', 'FECHA'])['cantidad_vendida'].sum().reset_index()

# Definir la fecha límite (1 de enero de 2023)
fecha_limite = pd.to_datetime('2023-01-01')

# Filtrar productos con ventas después de la fecha límite
productos_con_ventas = df_grouped[df_grouped['FECHA'] >= fecha_limite]['PRODUCTO'].unique()

# Filtrar el dataframe original para que solo contenga esos productos
df_filtrado = df_grouped[df_grouped['PRODUCTO'].isin(productos_con_ventas)]

def filter_products(df, min_samples=12):
    productos = df['PRODUCTO'].unique()
    productos_validos = []
    for producto in productos:
        df_producto = df[df['PRODUCTO'] == producto]
        if len(df_producto) >= min_samples:
            productos_validos.append(producto)
    return productos_validos

# Filtrar productos válidos
productos_validos = filter_products(df_filtrado, min_samples=12)

# **Condición temporal para limitar el análisis a 10 productos**
# **Comentario: Puedes eliminar la siguiente línea para realizar el análisis completo**
# productos_validos = productos_validos[:10]

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_lstm(df, producto, look_back=12):
    df_producto = df[df['PRODUCTO'] == producto].sort_values(by='FECHA')
    data = df_producto['cantidad_vendida'].values
    if len(data) < look_back + 1:
        return None, None, None  # Retorna None si no hay suficientes datos
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def prepare_data_ml(df, producto, look_back=12):
    df_producto = df[df['PRODUCTO'] == producto].sort_values(by='FECHA')
    data = df_producto['cantidad_vendida'].values
    if len(data) < look_back + 1:
        return None, None
    
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Crear un DataFrame vacío para almacenar las predicciones y el modelo seleccionado
resultados_predicciones = pd.DataFrame(columns=['PRODUCTO', 'DESCRIPCION', 'Fecha', 'Valor'])

# Obtener el total de productos a analizar
total_productos = len(productos_validos)
print(f"Total de productos a analizar: {total_productos}\n")

# Loop para procesar cada producto válido
for i, producto in enumerate(productos_validos, 1):
    try:
        descripcion = df_filtrado[df_filtrado['PRODUCTO'] == producto]['DESCRIPCION'].iloc[0]
        # Mostrar el nombre del producto que se está analizando junto con el contador y el porcentaje
        print(f"Analizando producto {i}/{total_productos} ({(i/total_productos)*100:.2f}%): {producto} - {descripcion}")
        
        # Preparar los datos para el producto actual
        X_lstm, y_lstm, scaler = prepare_data_lstm(df_filtrado, producto)
        X_ml, y_ml = prepare_data_ml(df_filtrado, producto)
        
        if X_lstm is None or y_lstm is None or X_ml is None or y_ml is None:
            continue
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
        
        # Entrenar y evaluar el modelo LSTM
        print(f"Evaluando modelo LSTM para el producto {producto}...")
        model_lstm = build_lstm_model(input_shape=(X_train_lstm.shape[1], 1))
        model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test_lstm), verbose=0)
        y_pred_lstm = model_lstm.predict(X_test_lstm)
        mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
        
        # Entrenar y evaluar el modelo de Regresión Lineal
        print(f"Evaluando modelo de Regresión Lineal para el producto {producto}...")
        model_lr = LinearRegression()
        model_lr.fit(X_train_ml, y_train_ml)
        y_pred_lr = model_lr.predict(X_test_ml)
        mse_lr = mean_squared_error(y_test_ml, y_pred_lr)
        
        # Entrenar y evaluar el modelo de Bosque Aleatorio
        print(f"Evaluando modelo de Bosque Aleatorio para el producto {producto}...")
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_train_ml, y_train_ml.ravel())
        y_pred_rf = model_rf.predict(X_test_ml)
        mse_rf = mean_squared_error(y_test_ml, y_pred_rf)
        
        # Seleccionar el mejor modelo basado en el MSE
        mse_dict = {'LSTM': mse_lstm, 'Linear Regression': mse_lr, 'Random Forest': mse_rf}
        best_model = min(mse_dict, key=mse_dict.get)
        
        # Mostrar tabla resumen
        resumen_df = pd.DataFrame({
            'Modelo': ['LSTM', 'Linear Regression', 'Random Forest'],
            'MSE': [mse_lstm, mse_lr, mse_rf]
        })
        resumen_df['Seleccionado'] = resumen_df['Modelo'] == best_model
        print(f"\nResumen del análisis para el producto {producto}:")
        print(resumen_df)
        
        # Predecir los próximos 18 meses con el mejor modelo
        if best_model == 'LSTM':
            predictions = []
            last_look_back = X_test_lstm[-1]
            for i in range(18):
                pred = model_lstm.predict(last_look_back.reshape(1, -1, 1))
                predictions.append(pred[0, 0])
                last_look_back = np.append(last_look_back[1:], pred).reshape(-1, 1)
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        elif best_model == 'Linear Regression':
            predictions = model_lr.predict(X_test_ml[-18:])
        elif best_model == 'Random Forest':
            predictions = model_rf.predict(X_test_ml[-18:])
        
        # Crear las fechas para los próximos 18 meses a partir de agosto 2024
        start_date = pd.to_datetime('2024-08-01')
        future_dates = [start_date + pd.DateOffset(months=i) for i in range(18)]
        
        # Convertir las fechas al formato YYYY-mm-dd
        future_dates = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        # Almacenar los resultados históricos en el DataFrame de resultados
        historico_df = df_filtrado[df_filtrado['PRODUCTO'] == producto][['FECHA', 'cantidad_vendida']].copy()
        historico_df['PRODUCTO'] = producto
        historico_df['DESCRIPCION'] = descripcion
        historico_df.rename(columns={'FECHA': 'Fecha', 'cantidad_vendida': 'Valor'}, inplace=True)
        resultados_predicciones = pd.concat([resultados_predicciones, historico_df])
        
        # Almacenar las predicciones en el DataFrame de resultados
        predicciones_df = pd.DataFrame({
            'PRODUCTO': producto,
            'DESCRIPCION': descripcion,
            'Fecha': future_dates,
            'Valor': predictions
        })
        resultados_predicciones = pd.concat([resultados_predicciones, predicciones_df])
        
        print(f"Predicciones generadas y almacenadas para el producto {producto}.")
    
    except Exception as e:
        print(f"Error procesando el producto {producto}: {str(e)}")
        continue

# Guardar los resultados en un archivo Excel
output_file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\predicciones_ventas.xlsx'
resultados_predicciones.to_excel(output_file_path, index=False)
print(f"Predicciones guardadas en {output_file_path}.")
