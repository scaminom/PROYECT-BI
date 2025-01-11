import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos procesados
print("Cargando datos...")
df = pd.read_csv('processed_order_profitability.csv')

# 2. Separar features y etiquetas
X = df.drop(['Profitability_Class', 'Profitability_Class_Encoded'], axis=1)
y = df['Profitability_Class_Encoded']

# 3. Aplicar t-SNE
print("Aplicando t-SNE (esto puede tomar algunos minutos)...")
tsne = TSNE(
    n_components=2,
    random_state=42,
    n_jobs=-1,  # Usar todos los núcleos disponibles
    perplexity=30,
    early_exaggeration=12
)
X_tsne = tsne.fit_transform(X)

# 4. Crear un DataFrame para la visualización
tsne_data = pd.DataFrame({
    'x': X_tsne[:, 0],
    'y': X_tsne[:, 1],
    'class': df['Profitability_Class']  # Usamos las etiquetas originales para mejor interpretación
})

# 5. Crear la visualización
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=tsne_data,
    x='x',
    y='y',
    hue='class',
    palette='deep',
    alpha=0.6
)

plt.title('Visualización t-SNE de los Datos de Rentabilidad')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Clase de Rentabilidad', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tsne_visualization.png')
plt.show()

print("Visualización completada y guardada como 'tsne_visualization.png'")