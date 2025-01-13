import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import preprocesar_datos, guardar_datos_preprocesados
import os

def analizar_datos(df):
    """
    Realiza el análisis exploratorio de datos para el dataset de demanda
    """
    print("\n=== Información básica del dataset ===")
    print(df.info())
    
    print("\n=== Primeras 5 filas ===")
    print(df.head())
    
    print("\n=== Análisis de valores nulos ===")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0])
    
    print("\n=== Estadísticas descriptivas ===")
    print(df.describe())
    
    # Distribución de temporadas
    plt.figure(figsize=(10, 6))
    df['Season'].value_counts().plot(kind='bar')
    plt.title('Distribución de Temporadas')
    plt.xlabel('Temporada')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Matriz de correlación
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación - Variables Numéricas')
    plt.tight_layout()
    plt.show()

    # Análisis de ventas por temporada
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Season', y='Total_Sales_Amount', data=df)
    plt.title('Distribución de Ventas por Temporada')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Análisis de cantidad vendida por mes
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Total_Quantity_Sold', data=df)
    plt.title('Cantidad Vendida por Mes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def verificar_preprocesamiento(resultados):
    """
    Verifica los resultados del preprocesamiento
    """
    print("\n=== Verificación del Preprocesamiento ===")
    
    # Distribución de temporadas en entrenamiento
    print("\nDistribución de temporadas en el conjunto de entrenamiento:")
    clases_unicas, conteos = np.unique(resultados['y_train'], return_counts=True)
    for clase, conteo in zip(clases_unicas, conteos):
        print(f"Temporada {clase}: {conteo} muestras")
    
    # Formas de los conjuntos
    print("\nFormas de los conjuntos de datos:")
    print(f"X_train: {resultados['X_train'].shape}")
    print(f"X_test: {resultados['X_test'].shape}")
    print(f"y_train: {resultados['y_train'].shape}")
    print(f"y_test: {resultados['y_test'].shape}")
    
    # Rango de valores después del escalamiento
    print("\nRango de valores después del escalamiento (X_train):")
    for columna in resultados['X_train'].columns:
        min_val = resultados['X_train'][columna].min()
        max_val = resultados['X_train'][columna].max()
        print(f"{columna}: [{min_val:.2f}, {max_val:.2f}]")

def main():
    """
    Función principal que ejecuta el análisis y preprocesamiento
    """
    try:
        # Cargar datos
        print("Cargando datos...")
        df = pd.read_csv('demanda_producto.csv', sep=';')
        
        # Realizar análisis exploratorio
        print("\nRealizando análisis exploratorio...")
        analizar_datos(df)
        
        # Realizar preprocesamiento
        print("\nRealizando preprocesamiento...")
        resultados = preprocesar_datos(df)
        
        # Verificar resultados del preprocesamiento
        verificar_preprocesamiento(resultados)
        
        # Guardar datos preprocesados
        print("\nGuardando datos preprocesados...")
        guardar_datos_preprocesados(resultados)
        
        print("\n¡Proceso completado con éxito!")
        
    except Exception as e:
        print(f"\nError durante el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()