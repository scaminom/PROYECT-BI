import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar datos
print("Cargando datos...")
df = pd.read_csv('processed_order_profitability.csv')

# 2. Preparar los datos
# Asegurarnos que los datos sean numéricos
X = df.drop(['Profitability_Class', 'Profitability_Class_Encoded'], axis=1).astype('float32')
y = df['Profitability_Class_Encoded'].astype('int32')

# 3. Dividir datos
print("Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Convertir a arrays de numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 5. Calcular pesos de clase
print("Calculando pesos de clase...")
weights = compute_class_weight('balanced', 
                             classes=np.unique(y_train), 
                             y=y_train)
class_weights = dict(zip(range(len(weights)), weights))

# 6. Crear el modelo
print("Creando modelo...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 7. Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 9. Entrenar
print("\nEntrenando modelo...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

# 10. Evaluar
print("\nEvaluando modelo...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en conjunto de prueba: {test_accuracy:.4f}")

# 11. Predecir y crear matriz de confusión
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 12. Visualizar resultados
plt.figure(figsize=(12, 5))

# Gráfica de accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 13. Imprimir reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred_classes,
    target_names=['Alta_Rentabilidad', 'Baja_Rentabilidad', 'Perdida', 'Rentabilidad_Media']
))