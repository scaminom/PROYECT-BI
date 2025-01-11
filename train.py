import numpy as np
from utils.preprocessing import cargar_datos_preprocesados
from utils.model import (crear_modelo, obtener_callbacks, 
                        evaluar_modelo, visualizar_entrenamiento,
                        guardar_modelo)

def entrenar():
    """
    Función principal de entrenamiento
    """
    try:
        # Cargar datos preprocesados
        print("Cargando datos preprocesados...")
        datos = cargar_datos_preprocesados()
        
        # Verificar la forma de los datos
        print(f"Forma de X_train: {datos['X_train'].shape}")
        print(f"Forma de y_train: {datos['y_train'].shape}")
        
        # Crear modelo
        print("\nCreando modelo...")
        model = crear_modelo(datos['X_train'].shape[1])
        
        # Obtener callbacks
        callbacks = obtener_callbacks()
        
        # Entrenar modelo
        print("\nIniciando entrenamiento...")
        history = model.fit(
            datos['X_train'], 
            datos['y_train'],
            validation_data=(datos['X_test'], datos['y_test']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        print("\nEvaluando modelo...")
        evaluar_modelo(model, datos['X_test'], datos['y_test'], 
                      datos['label_encoder_y'])
        
        # Visualizar métricas de entrenamiento
        print("\nVisualizando métricas de entrenamiento...")
        visualizar_entrenamiento(history)
        
        # Guardar modelo
        print("\nGuardando modelo...")
        guardar_modelo(model)
        
        print("\n¡Entrenamiento completado con éxito!")
        
        return model, history
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    model, history = entrenar()