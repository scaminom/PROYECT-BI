import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocesar_datos(df):
    """
    Pipeline completo de preprocesamiento de datos para predicción de demanda.
    """
    # 1. Crear una copia para no modificar los datos originales
    df_proc = df.copy()
    
    # 2. Manejo de valores nulos
    # Valores nulos en columnas categóricas
    df_proc['Size'] = df_proc['Size'].fillna('Sin Talla')
    
    # 3. Tratamiento de outliers usando IQR
    columnas_outliers = ['Total_Profit']
    
    for columna in columnas_outliers:
        Q1 = df_proc[columna].quantile(0.25)
        Q3 = df_proc[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_proc[columna] = df_proc[columna].clip(limite_inferior, limite_superior)
    
    # 4. Encoding de variables categóricas
    label_encoders = {}
    columnas_categoricas = ['Size', 'Month', 'State Province']
    
    for columna in columnas_categoricas:
        label_encoders[columna] = LabelEncoder()
        df_proc[columna] = label_encoders[columna].fit_transform(df_proc[columna])
    
    # 5. Separar features y target
    y = df_proc['Season']
    X = df_proc.drop(['Season'], axis=1)
    
    # 6. Escalamiento robusto para variables numéricas
    columnas_numericas = ['Total_Profit', 'Avg_Monthly_Sales', 'Calendar Month Number',
                          'Unique_Customers']
    
    scaler = RobustScaler()
    X[columnas_numericas] = scaler.fit_transform(X[columnas_numericas])
    
    # 7. Codificar variable objetivo
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    
    # 8. División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Verificar que no haya valores nulos antes de SMOTE
    print("\nVerificando valores nulos antes de SMOTE:")
    nulos = X_train.isnull().sum()
    if nulos.sum() > 0:
        print("Columnas con valores nulos:")
        print(nulos[nulos > 0])
        raise ValueError("Hay valores nulos en los datos. Por favor, revisar el preprocesamiento.")
    
    # 9. Balanceo de clases
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    # 10. Convertir a DataFrame
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
    
    return {
        'X_train': X_train_balanced,
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'label_encoder_y': label_encoder_y,
        'scaler': scaler
    }

def guardar_datos_preprocesados(resultados, ruta_base='datos/'):
    """
    Guarda los datos preprocesados y los encoders, incluyendo archivos CSV.
    Opcionalmente aplica ruido a los datos de entrenamiento.
    
    Args:
        resultados: Diccionario con los datos preprocesados
        ruta_base: Ruta donde guardar los archivos
    """
    # Crear directorio si no existe
    os.makedirs(ruta_base, exist_ok=True)
   
    # Guardar datos de entrenamiento y prueba en formato numpy
    np.save(f'{ruta_base}X_train.npy', resultados['X_train'])
    np.save(f'{ruta_base}X_test.npy', resultados['X_test'])
    np.save(f'{ruta_base}y_train.npy', resultados['y_train'])
    np.save(f'{ruta_base}y_test.npy', resultados['y_test'])
    
    # Guardar encoders y scaler
    with open(f'{ruta_base}preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'label_encoders': resultados['label_encoders'],
            'label_encoder_y': resultados['label_encoder_y'],
            'scaler': resultados['scaler']
        }, f)
    
    # Guardar datos en formato CSV
    df_train = pd.DataFrame(resultados['X_train'], columns=resultados['X_train'].columns)
    df_train['Season'] = resultados['y_train']
    df_train.to_csv(f'{ruta_base}train_processed.csv', index=False)
    
    # Test
    df_test = pd.DataFrame(resultados['X_test'], columns=resultados['X_test'].columns)
    df_test['Season'] = resultados['y_test']
    df_test.to_csv(f'{ruta_base}test_processed.csv', index=False)

def cargar_datos_preprocesados(ruta_base='datos/'):
    """
    Carga los datos preprocesados y los encoders
    """
    # Cargar datos y convertirlos a arrays de numpy
    X_train = pd.DataFrame(np.load(f'{ruta_base}X_train.npy'))
    X_test = pd.DataFrame(np.load(f'{ruta_base}X_test.npy'))
    y_train = np.load(f'{ruta_base}y_train.npy')
    y_test = np.load(f'{ruta_base}y_test.npy')
    
    # Cargar encoders y scaler
    with open(f'{ruta_base}preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    
    return {
        'X_train': X_train.values,
        'X_test': X_test.values,
        'y_train': y_train,
        'y_test': y_test,
        **preprocessors
    }