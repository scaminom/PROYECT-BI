import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocesar_datos(df):
    """
    Pipeline completo de preprocesamiento de datos para predicción de demanda.
    Versión actualizada usando One-Hot Encoding para todas las variables categóricas.
    """
    print("\n=== Iniciando preprocesamiento de datos ===")
    
    # 1. Crear una copia para no modificar los datos originales
    df_proc = df.copy()
    
    # 2. Verificar y mostrar información inicial
    print("\nInformación inicial del dataset:")
    print(f"Dimensiones: {df_proc.shape}")
    print("\nDistribución inicial de temporadas:")
    print(df_proc['Season'].value_counts())
    
    # 3. Manejo de valores nulos
    print("\nTratamiento de valores nulos...")
    columnas_categoricas = [col for col in df_proc.columns if df_proc[col].dtype == 'object']
    
    for columna in columnas_categoricas:
        if df_proc[columna].isnull().any():
            moda = df_proc[columna].mode()[0]
            df_proc[columna] = df_proc[columna].fillna(moda)
            print(f"Valores nulos en {columna} rellenados con: {moda}")
    
    # 4. Separar columnas numéricas y categóricas
    print("\nSeparando columnas numéricas y categóricas...")
    columnas_numericas = df_proc.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = [col for col in columnas_categoricas if col != 'Season']
    
    # 5. One-Hot Encoding para todas las variables categóricas
    print("\nAplicando One-Hot Encoding a variables categóricas...")
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Aplicar One-Hot Encoding
    if columnas_categoricas:
        onehot_features = onehot.fit_transform(df_proc[columnas_categoricas])
        
        # Crear nombres de columnas para one-hot
        onehot_columns = []
        for i, columna in enumerate(columnas_categoricas):
            cats = onehot.categories_[i]
            onehot_columns.extend([f"{columna}_{cat}" for cat in cats])
        
        # Convertir a DataFrame
        onehot_df = pd.DataFrame(
            onehot_features, 
            columns=onehot_columns,
            index=df_proc.index
        )
        
        # Eliminar columnas originales y añadir one-hot
        df_proc = df_proc.drop(columnas_categoricas, axis=1)
        df_proc = pd.concat([df_proc, onehot_df], axis=1)
        print(f"One-hot encoding aplicado a: {', '.join(columnas_categoricas)}")
    
    # 6. Escalamiento robusto para variables numéricas
    print("\nAplicando escalamiento a variables numéricas...")
    scaler = None
    if len(columnas_numericas) > 0:
        scaler = RobustScaler()
        df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])
        print(f"Escalamiento aplicado a: {len(columnas_numericas)} columnas")
    
    # 7. Codificar variable objetivo (Season)
    print("\nCodificando variable objetivo...")
    label_encoder_y = LabelEncoder()
    season_codes = label_encoder_y.fit_transform(df_proc['Season'])
    print("Mapeo de temporadas:", 
          dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_))))
    
    # 8. Separar features y target para splits
    print("\nSeparando features y target...")
    y = season_codes
    X = df_proc.drop(['Season'], axis=1)
    
    # 9. División train-test
    print("\nRealizando división train-test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n=== Preprocesamiento completado ===")
    print(f"Dimensiones finales:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    return {
        'df_completo': df_proc,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'onehot_encoder': onehot,
        'label_encoder_y': label_encoder_y,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }

def guardar_datos_preprocesados(resultados, ruta_base='datos/'):
    """
    Guarda los datos preprocesados y los encoders.
    """
    # Crear directorio si no existe
    os.makedirs(ruta_base, exist_ok=True)
    
    print("Guardando datos principales...")
    # Guardar datos principales
    np.save(f'{ruta_base}X_train.npy', resultados['X_train'])
    np.save(f'{ruta_base}X_test.npy', resultados['X_test'])
    np.save(f'{ruta_base}y_train.npy', resultados['y_train'])
    np.save(f'{ruta_base}y_test.npy', resultados['y_test'])
    
    print("Guardando encoders...")
    # Guardar encoders y nombres de características
    with open(f'{ruta_base}preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'onehot_encoder': resultados['onehot_encoder'],
            'label_encoder_y': resultados['label_encoder_y'],
            'scaler': resultados['scaler'],
            'feature_names': resultados['feature_names']
        }, f)
    
    print("Guardando archivos CSV...")
    # Guardar datos en formato CSV
    df_train = pd.DataFrame(resultados['X_train'], columns=resultados['feature_names'])
    df_train['Season'] = resultados['y_train']
    df_train.to_csv(f'{ruta_base}train_processed.csv', index=False)
    
    df_test = pd.DataFrame(resultados['X_test'], columns=resultados['feature_names'])
    df_test['Season'] = resultados['y_test']
    df_test.to_csv(f'{ruta_base}test_processed.csv', index=False)
    
    # Guardar dataset completo procesado
    resultados['df_completo'].to_csv(f'{ruta_base}dataset_procesado.csv', index=False)
    
    print("Guardado completado exitosamente en:", os.path.abspath(ruta_base))
    print("\nArchivos guardados:")
    for archivo in os.listdir(ruta_base):
        print(f"- {archivo}")

def cargar_datos_preprocesados(ruta_base='datos/'):
    """
    Carga los datos preprocesados y los encoders.
    """
    print("\n=== Cargando datos preprocesados ===")
    try:
        # Cargar datos
        print("Cargando arrays numpy...")
        X_train = np.load(f'{ruta_base}X_train.npy')
        X_test = np.load(f'{ruta_base}X_test.npy')
        y_train = np.load(f'{ruta_base}y_train.npy')
        y_test = np.load(f'{ruta_base}y_test.npy')
        
        print("Cargando encoders y nombres de características...")
        # Cargar encoders y nombres de características
        with open(f'{ruta_base}preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
        
        print("Carga completada exitosamente")
        print(f"\nDimensiones de los datos:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            **preprocessors
        }
    
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        raise