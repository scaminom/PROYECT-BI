import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

def cargar_datos(ruta_archivo: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga y prepara los datos desde un archivo CSV.
    
    Args:
        ruta_archivo: Ruta al archivo CSV
    
    Returns:
        Tupla con features (X) y etiquetas (y)
    """
    df = pd.read_csv(ruta_archivo, sep=';')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def preprocesar_datos(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesa los datos, incluyendo escalamiento y división en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Features
        y: Etiquetas
    
    Returns:
        Tupla con datos de entrenamiento y prueba procesados
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def crear_modelo_softmax(input_dim: int, num_clases: int) -> tf.keras.Model:
    """
    Crea un modelo de regresión logística multinomial (softmax).
    
    Args:
        input_dim: Dimensión de entrada
        num_clases: Número de clases para la salida
    
    Returns:
        Modelo de TensorFlow compilado
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(num_clases, activation='softmax', input_dim=input_dim)
    ])
    
    modelo.compile(
        optimizer='adam', # 0.001 learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

def entrenar_modelo(
    modelo: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    """
    Entrena el modelo y retorna el historial de entrenamiento.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )
    
    return history

def visualizar_entrenamiento(history: tf.keras.callbacks.History) -> None:
    """
    Visualiza las métricas de entrenamiento.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Entrenamiento')
    ax1.plot(history.history['val_loss'], label='Validación')
    ax1.set_title('Pérdida del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Entrenamiento')
    ax2.plot(history.history['val_accuracy'], label='Validación')
    ax2.set_title('Precisión del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main(ruta_archivo: str) -> None:
    """
    Función principal que ejecuta todo el proceso.
    """
    print("Cargando datos...")
    X, y = cargar_datos(ruta_archivo)
    
    print("Preprocesando datos...")
    X_train, X_test, y_train, y_test = preprocesar_datos(X, y)
    
    print("\nCreando modelo softmax...")
    num_clases = len(np.unique(y))
    modelo = crear_modelo_softmax(X_train.shape[1], num_clases)
    
    modelo.summary()
    
    print("\nEntrenando modelo...")
    history = entrenar_modelo(modelo, X_train, y_train, X_test, y_test)
    
    print("\nEvaluando modelo...")
    test_loss, test_accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en conjunto de prueba: {test_accuracy:.4f}")
    
    print("\nVisualizando resultados del entrenamiento...")
    visualizar_entrenamiento(history)

if __name__ == "__main__":
    ruta_archivo = "dataset-onehotencoding.csv"
    main(ruta_archivo)