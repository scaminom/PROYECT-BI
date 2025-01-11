import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

# 1. Cargar los datos procesados
df = pd.read_csv('processed_order_profitability.csv')

# 2. Separar características (X) y etiquetas (y)
X = df.drop(['Profitability_Class', 'Profitability_Class_Encoded'], axis=1)
y = df['Profitability_Class_Encoded']

# 3. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Calcular los pesos de las clases - MODIFICADO
unique_classes = np.sort(np.unique(y_train))  # Aseguramos que las clases estén ordenadas
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)
class_weight_dict = dict(zip(unique_classes, class_weights))

# 5. Crear el modelo - MODIFICADO
input_layer = keras.layers.Input(shape=(X_train.shape[1],))
x = keras.layers.Dense(128, activation='relu')(input_layer)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.1)(x)
output_layer = keras.layers.Dense(len(unique_classes), activation='softmax')(x)

model = keras.Model(inputs=input_layer, outputs=output_layer)

# 6. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=5,
        monitor='val_loss',
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        monitor='val_loss'
    )
]

# 8. Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=512,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 9. Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'\nPrecisión en el conjunto de prueba: {test_accuracy:.4f}')

# 10. Hacer predicciones y mostrar métricas detalladas
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nReporte de clasificación:")
print(classification_report(
    y_test, 
    y_pred_classes, 
    target_names=['Alta_Rentabilidad', 'Baja_Rentabilidad', 'Perdida', 'Rentabilidad_Media']
))

# 11. Guardar el modelo
model.save('profitability_model.h5')

# 12. Visualizar el entrenamiento
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Curvas de pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Curvas de precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()