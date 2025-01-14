import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocesar_datos(df):
    """
    Pipeline completo de preprocesamiento de datos para predicción de demanda.
    Versión actualizada compatible con versiones modernas de las bibliotecas.
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
    # Valores nulos en columnas categóricas
    columnas_categoricas = [col for col in df_proc.columns if df_proc[col].dtype == 'object']
    
    for columna in columnas_categoricas:
        if df_proc[columna].isnull().any():
            moda = df_proc[columna].mode()[0]
            df_proc[columna] = df_proc[columna].fillna(moda)
            print(f"Valores nulos en {columna} rellenados con: {moda}")
    
    # 4. Encoding de variables categóricas
    print("\nAplicando encoding a variables categóricas...")
    
    # Inicializar diccionarios para encoders
    label_encoders = {}
    
    # Identificar columnas para diferentes tipos de encoding 
    columnas_onehot = ['Thickness', 'Weather_Resistance', 'Target_Gender',
                      'Price_Category', 'Fit_Type']
    columnas_label = [col for col in columnas_categoricas
                     if col != 'Season' and col not in columnas_onehot]
    
    # Label Encoding para la mayoría de las variables categóricas
    for columna in columnas_label:
        if columna in df_proc.columns:
            label_encoders[columna] = LabelEncoder()
            df_proc[columna] = label_encoders[columna].fit_transform(df_proc[columna])
            print(f"Label encoding aplicado a: {columna}") 
    
    # One-Hot Encoding para variables categóricas seleccionadas
    columnas_onehot_presentes = [col for col in columnas_onehot if col in df_proc.columns]
    if columnas_onehot_presentes:
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
        onehot_features = onehot.fit_transform(df_proc[columnas_onehot_presentes])
        
        # Crear nombres de columnas para one-hot
        onehot_columns = []
        for i, columna in enumerate(columnas_onehot_presentes):
            cats = onehot.categories_[i]
            onehot_columns.extend([f"{columna}_{cat}" for cat in cats])
        
        # Convertir a DataFrame y concatenar 
        onehot_df = pd.DataFrame(onehot_features, columns=onehot_columns,
                               index=df_proc.index)
        
        # Eliminar columnas originales y añadir one-hot 
        df_proc = df_proc.drop(columnas_onehot_presentes, axis=1)
        df_proc = pd.concat([df_proc, onehot_df], axis=1)
        print(f"One-hot encoding aplicado a: {', '.join(columnas_onehot_presentes)}") 
    
    # 5. Escalamiento robusto para variables numéricas
    print("\nAplicando escalamiento a variables numéricas...")
    columnas_numericas = df_proc.select_dtypes(include=['int64', 'float64']).columns
    columnas_numericas = columnas_numericas.drop('Season_Code') if 'Season_Code' in columnas_numericas else columnas_numericas
    
    if len(columnas_numericas) > 0:
        scaler = RobustScaler()  
        df_proc[columnas_numericas] = scaler.fit_transform(df_proc[columnas_numericas])
        print(f"Escalamiento aplicado a: {len(columnas_numericas)} columnas")
    
    # 6. Codificar variable objetivo para splits
    print("\nCodificando variable objetivo...") 
    label_encoder_y = LabelEncoder()
    df_proc['Season_Code'] = label_encoder_y.fit_transform(df_proc['Season'])
    print("Mapeo de temporadas:",
          dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_))))
    
    print("\n=== Preprocesamiento completado ===")
    print(f"\nDimensiones finales del dataset procesado completo: {df_proc.shape}")

    # 7. Separar features y target para splits
    print("\nSeparando features y target...")
    y = df_proc['Season_Code'] 
    X = df_proc.drop(['Season', 'Season_Code'], axis=1)
    
    # 8. División train-test  
    print("\nRealizando división train-test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 9. Balanceo de clases
    print("\nAplicando balanceo de clases...")
    print("Distribución antes del balanceo:")  
    print(pd.Series(y_train).value_counts())
    
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    print("\nDistribución después del balanceo:")
    print(pd.Series(y_train_balanced).value_counts())

    # 10. Conversión final a DataFrame
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
    
    print("\n=== Preprocesamiento y balanceo completados ===") 
    print(f"Dimensiones finales:")
    print(f"X_train: {X_train_balanced.shape}") 
    print(f"X_test: {X_test.shape}")
    
    return {
        'df_completo': df_proc,
        'X_train': X_train_balanced, 
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_test': y_test, 
        'label_encoders': label_encoders,
        'label_encoder_y': label_encoder_y, 
        'scaler': scaler if len(columnas_numericas) > 0 else None
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
    # Guardar encoders
    with open(f'{ruta_base}preprocessors.pkl', 'wb') as f:
        pickle.dump({
            'label_encoders': resultados['label_encoders'], 
            'label_encoder_y': resultados['label_encoder_y'],
            'scaler': resultados['scaler'] 
        }, f)
    
    print("Guardando archivos CSV...")  
    # Guardar datos en formato CSV
    df_train = pd.DataFrame(resultados['X_train'], columns=resultados['X_train'].columns)
    df_train['Season'] = resultados['y_train']
    df_train.to_csv(f'{ruta_base}train_processed.csv', index=False)
    
    df_test = pd.DataFrame(resultados['X_test'], columns=resultados['X_test'].columns) 
    df_test['Season'] = resultados['y_test']
    df_test.to_csv(f'{ruta_base}test_processed.csv', index=False)

    # Guardar dataset completo procesado con Season sin codificar
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
        X_train = pd.DataFrame(np.load(f'{ruta_base}X_train.npy'))
        X_test = pd.DataFrame(np.load(f'{ruta_base}X_test.npy'))
        y_train = np.load(f'{ruta_base}y_train.npy')
        y_test = np.load(f'{ruta_base}y_test.npy')
        
        print("Cargando encoders...")
        # Cargar encoders
        with open(f'{ruta_base}preprocessors.pkl', 'rb') as f:
            preprocessors = pickle.load(f)
        
        print("Carga completada exitosamente")
        print(f"\nDimensiones de los datos:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'y_train': y_train,
            'y_test': y_test,
            **preprocessors
        }
    
    except Exception as e:
        print(f"Error al cargar los datos: {str(e)}")
        raise