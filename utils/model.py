import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def crear_modelo(input_shape):
    """
    Crea el modelo de red neuronal
    """
    model = Sequential([
        # Capa de entrada
        Dense(64, input_dim=input_shape, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Primera capa oculta
        Dense(32, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Segunda capa oculta
        Dense(16, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        
        # Capa de salida
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Quitar AUC por ahora para simplificar
    )
    
    return model

def obtener_callbacks():
    """
    Retorna los callbacks para el entrenamiento
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]

def evaluar_modelo(model, X_test, y_test, label_encoder_y):
    """
    Evalúa el modelo y muestra métricas
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Reporte de clasificación
    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder_y.classes_))
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder_y.classes_,
                yticklabels=label_encoder_y.classes_)
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()

def visualizar_entrenamiento(history):
    """
    Visualiza las métricas de entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de pérdida
    ax1.plot(history.history['loss'], label='Entrenamiento')
    ax1.plot(history.history['val_loss'], label='Validación')
    ax1.set_title('Pérdida del Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    
    # Gráfico de accuracy
    ax2.plot(history.history['accuracy'], label='Entrenamiento')
    ax2.plot(history.history['val_accuracy'], label='Validación')
    ax2.set_title('Accuracy del Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def guardar_modelo(model, ruta='modelos/modelo.h5'):
    """
    Guarda el modelo entrenado
    """
    import os
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    model.save(ruta)

def cargar_modelo(ruta='modelos/modelo.h5'):
    """
    Carga un modelo guardado
    """
    return tf.keras.models.load_model(ruta)