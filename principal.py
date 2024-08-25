import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df.rename(columns={'Month': 'MES', 'CVE PRODUCTO': 'PRODUCTO', 'LBS': 'cantidad_vendida'}, inplace=True)
    df['FECHA'] = pd.to_datetime(df['AÑO'].astype(str) + '-' + df['MES'].astype(str) + '-01')
    df_grouped = df.groupby(['PRODUCTO', 'DESCRIPCION', 'FECHA', 'CONV/ORG'])['cantidad_vendida'].sum().reset_index()
    fecha_limite = pd.to_datetime('2023-01-01')
    productos_con_ventas = df_grouped[df_grouped['FECHA'] >= fecha_limite]['PRODUCTO'].unique()
    df_filtrado = df_grouped[df_grouped['PRODUCTO'].isin(productos_con_ventas)]
    return df_filtrado

def filter_products(df, min_samples=12):
    productos = df['PRODUCTO'].unique()
    productos_validos = [producto for producto in productos if len(df[df['PRODUCTO'] == producto]) >= min_samples]
    return productos_validos

def prepare_data_lstm(df, producto, look_back=12):
    df_producto = df[df['PRODUCTO'] == producto].sort_values(by='FECHA')
    data = df_producto['cantidad_vendida'].values
    if len(data) < look_back + 1:
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
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

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def analyze_products(df_filtrado, productos_validos):
    resultados_predicciones = pd.DataFrame(columns=['PRODUCTO', 'DESCRIPCION', 'Fecha', 'Valor', 'CONV/ORG'])
    total_productos = len(productos_validos)
    print(f"Total de productos a analizar: {total_productos}\n")

    for i, producto in enumerate(productos_validos, 1):
        try:
            descripcion = df_filtrado[df_filtrado['PRODUCTO'] == producto]['DESCRIPCION'].iloc[0]
            conv_org = df_filtrado[df_filtrado['PRODUCTO'] == producto]['CONV/ORG'].iloc[0]
            print(f"Analizando producto {i}/{total_productos} ({(i/total_productos)*100:.2f}%): {producto} - {descripcion}")
            
            X_lstm, y_lstm, scaler = prepare_data_lstm(df_filtrado, producto)
            X_ml, y_ml = prepare_data_ml(df_filtrado, producto)
            
            if X_lstm is None or y_lstm is None or X_ml is None or y_ml is None:
                continue
            
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
            X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            
            print(f"Evaluando modelo LSTM para el producto {producto}...")
            model_lstm = build_lstm_model(input_shape=(X_train_lstm.shape[1], 1))
            model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test_lstm), verbose=0)
            y_pred_lstm = model_lstm.predict(X_test_lstm)
            mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
            
            print(f"Evaluando modelo de Regresión Lineal para el producto {producto}...")
            model_lr = SGDRegressor(max_iter=1000, tol=1e-3)
            model_lr.fit(X_train_ml, y_train_ml)
            y_pred_lr = model_lr.predict(X_test_ml)
            mse_lr = mean_squared_error(y_test_ml, y_pred_lr)
            
            print(f"Evaluando modelo de Bosque Aleatorio para el producto {producto}...")
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            model_rf.fit(X_train_ml, y_train_ml.ravel())
            y_pred_rf = model_rf.predict(X_test_ml)
            mse_rf = mean_squared_error(y_test_ml, y_pred_rf)
            
            mse_dict = {'LSTM': mse_lstm, 'Linear Regression': mse_lr, 'Random Forest': mse_rf}
            best_model = min(mse_dict, key=mse_dict.get)
            
            resumen_df = pd.DataFrame({
                'Modelo': ['LSTM', 'Linear Regression', 'Random Forest'],
                'MSE': [mse_lstm, mse_lr, mse_rf]
            })
            resumen_df['Seleccionado'] = resumen_df['Modelo'] == best_model
            print(f"\nResumen del análisis para el producto {producto}:")
            print(resumen_df)
            
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
            
            start_date = pd.to_datetime('2024-08-01')
            future_dates = [start_date + pd.DateOffset(months=i) for i in range(18)]
            future_dates = [d.strftime('%Y-%m-%d') for d in future_dates]
            
            historico_df = df_filtrado[df_filtrado['PRODUCTO'] == producto][['FECHA', 'cantidad_vendida', 'CONV/ORG']].copy()
            historico_df['PRODUCTO'] = producto
            historico_df['DESCRIPCION'] = descripcion
            historico_df.rename(columns={'FECHA': 'Fecha', 'cantidad_vendida': 'Valor'}, inplace=True)
            resultados_predicciones = pd.concat([resultados_predicciones, historico_df])
            
            predicciones_df = pd.DataFrame({
                'PRODUCTO': producto,
                'DESCRIPCION': descripcion,
                'Fecha': future_dates,
                'Valor': predictions,
                'CONV/ORG': conv_org
            })
            resultados_predicciones = pd.concat([resultados_predicciones, predicciones_df])
            
            print(f"Predicciones generadas y almacenadas para el producto {producto}.")
        
        except Exception as e:
            print(f"Error procesando el producto {producto}: {str(e)}")
            continue
    
    return resultados_predicciones

def save_results(resultados_predicciones, output_file_path):
    resultados_predicciones.to_excel(output_file_path, index=False)
    print(f"Predicciones guardadas en {output_file_path}.")

def pivot_and_save(df, output_file_path):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.strftime('%Y-%m')
    df_pivot = df.pivot_table(index=['PRODUCTO', 'DESCRIPCION', 'CONV/ORG'], columns='Mes', values='Valor', aggfunc='sum')
    df_pivot.reset_index(inplace=True)
    df_pivot.rename(columns={'Valor': 'LBS'}, inplace=True)
    df_pivot.to_excel(output_file_path, index=False)
    print(f"DataFrame pivoteado guardado en {output_file_path}.")

def update_model_with_new_data(historical_file_path, new_data_file_path):
    # Cargar datos históricos y nuevos datos
    df_historical = pd.read_csv(historical_file_path, encoding='utf-8')
    df_new_data = pd.read_csv(new_data_file_path, encoding='utf-8')

    # Mostrar la secuencia de datos leídos
    print("Datos históricos leídos:")
    print(df_historical.head())
    print("\nNuevos datos leídos:")
    print(df_new_data.head())

    # Concatenar los datos
    df_combined = pd.concat([df_historical, df_new_data]).drop_duplicates().reset_index(drop=True)

    # Eliminar los registros cuyo valor de la columna LBS sea cero
    df_combined = df_combined[df_combined['LBS'] != 0]
    
    # Guardar el archivo actualizado
    df_combined.to_csv(historical_file_path, index=False, encoding='utf-8')
    print(f"Archivo histórico actualizado con nuevos datos y guardado en {historical_file_path}.")

    return df_combined

# Rutas de archivos
historical_file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\historial ventas.csv'
new_data_file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\historial ventas actualizado.csv'
output_file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\predicciones_ventas.xlsx'
pivot_output_file_path = r'C:\Users\agomez\OneDrive - MarBran SA de CV\1.4 ANALISIS VENTAS IA\predicciones_ventas_pivoteado.xlsx'

# Actualizar el modelo con nuevos datos
df_combined = update_model_with_new_data(historical_file_path, new_data_file_path)

# Ejecución del flujo de trabajo
df_filtrado = load_and_preprocess_data(historical_file_path)
productos_validos = filter_products(df_filtrado, min_samples=12)
resultados_predicciones = analyze_products(df_filtrado, productos_validos)
save_results(resultados_predicciones, output_file_path)

# Leer el archivo de Excel y pivotear
df = pd.read_excel(output_file_path)
pivot_and_save(df, pivot_output_file_path)