import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Cargar los datos
df = pd.read_csv('order_profitability.csv')

# 2. Definir las columnas por tipo
numeric_features = [
    'Lead Time Days', 
    'Is Chiller Stock',
    'Typical Weight Per Unit',
    'Product_Base_Price',
    'Month',
    'Fiscal Year',
    'Is_Holiday_Season',
    'Quantity',
    'Tax Rate',
    'Profit_Margin'
]

categorical_features = [
    'Buying Group',
    'Customer_Category',
    'Sales Territory',
    'Continent',
    'Country',
    'Package'
]

# 3. Manejar valores faltantes
# Para características numéricas
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

# Para características categóricas
categorical_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

# 4. Normalizar características numéricas
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 5. Codificar variables categóricas
encoded_features = pd.DataFrame()  # DataFrame para almacenar las características codificadas

for feature in categorical_features:
    # Crear dummies y excluir una categoría (drop_first=True)
    dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
    encoded_features = pd.concat([encoded_features, dummies], axis=1)

# 6. Codificar la variable objetivo (Profitability_Class)
label_encoder = LabelEncoder()
df['Profitability_Class_Encoded'] = label_encoder.fit_transform(df['Profitability_Class'])

# 7. Crear el DataFrame final
final_df = pd.concat([
    df[numeric_features],  # Características numéricas normalizadas
    encoded_features,      # Características categóricas codificadas
    df['Profitability_Class'],  # Etiqueta original
    df['Profitability_Class_Encoded']  # Etiqueta codificada
], axis=1)

# 8. Guardar el mapeo de las etiquetas para referencia futura
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapeo de etiquetas:", label_mapping)

# 9. Guardar los datos procesados
final_df.to_csv('processed_order_profitability.csv', index=False)

# 10. Mostrar información sobre los datos procesados
print("\nForma del dataset procesado:", final_df.shape)
print("\nColumnas en el dataset procesado:")
print(final_df.columns.tolist())
print("\nDistribución de clases:")
print(final_df['Profitability_Class'].value_counts())