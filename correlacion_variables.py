import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar los datos
print("Cargando datos...")
df = pd.read_csv('processed_order_profitability.csv')

# 2. Seleccionar solo las columnas numéricas
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# 3. Calcular la matriz de correlación
correlation_matrix = df_numeric.corr()

# 4. Crear visualización de la matriz de correlación
plt.figure(figsize=(20, 16))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    linewidths=0.5,
    square=True
)

plt.title('Matriz de Correlación de Variables Numéricas', pad=20, size=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Identificar correlaciones altas (>0.7 o <-0.7)
print("\nCorrelaciones significativas (|corr| > 0.7):")
high_corr = np.where(np.abs(correlation_matrix) > 0.7)
high_corr = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y]) 
             for x, y in zip(*high_corr) if x != y]

for var1, var2, corr in high_corr:
    print(f"{var1} - {var2}: {corr:.3f}")

# 6. Analizar correlaciones con la variable objetivo (Profitability_Class_Encoded)
target_corr = correlation_matrix['Profitability_Class_Encoded'].sort_values(ascending=False)
print("\nCorrelaciones con Profitability_Class_Encoded:")
print(target_corr)

# 7. Guardar resultados
with open('correlation_analysis.txt', 'w') as f:
    f.write("Análisis de Correlación\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Correlaciones significativas (|corr| > 0.7):\n")
    for var1, var2, corr in high_corr:
        f.write(f"{var1} - {var2}: {corr:.3f}\n")
    
    f.write("\nCorrelaciones con Profitability_Class_Encoded:\n")
    f.write(target_corr.to_string())

print("\nAnálisis completo guardado en 'correlation_analysis.txt'")