import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import chi2_contingency

def calculate_cramers_v(x, y):
    """
    Calcula el coeficiente V de Cramér para variables categóricas
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

def corregir_combinacion(row):
    """
    Corrige combinaciones inválidas en los datos
    """
    if row['Season'] == 'Summer' and row['Thermal_Rating'] in ['Muy Alto', 'Alto']:
        return pd.Series({
            'Season': 'Winter',
            'Thermal_Rating': row['Thermal_Rating'],
            'Thickness': 'Grueso'
        })
    elif row['Season'] == 'Winter' and row['Product_Category'] == 'Swimwear':
        return pd.Series({
            'Season': 'Summer',
            'Thermal_Rating': 'Bajo',
            'Thickness': 'Ligero'
        })
    return row

def generar_dataset_completo(n_samples=95060, random_state=42):
    """
    Genera un dataset completo con features más relevantes y mejor separación entre clases.
    """
    np.random.seed(random_state)
    
    # 1. Características base del producto con pesos más diferenciados
    product_categories = {
        'Outerwear': {
            'items': ['Chaquetas', 'Abrigos', 'Impermeables', 'Chalecos'],
            'season_weights': {
                'Winter': 0.90,
                'Fall': 0.08,
                'Spring': 0.02,
                'Summer': 0.00
            }
        },
        'Tops': {
            'items': ['Camisetas', 'Blusas', 'Polos', 'Sudaderas'],
            'season_weights': {
                'Winter': 0.05,
                'Fall': 0.15,
                'Spring': 0.35,
                'Summer': 0.45
            }
        },
        'Bottoms': {
            'items': ['Pantalones', 'Shorts', 'Faldas', 'Jeans'],
            'season_weights': {
                'Winter': 0.45,
                'Fall': 0.35,
                'Spring': 0.15,
                'Summer': 0.05
            }
        },
        'Dresses': {
            'items': ['Vestidos Casual', 'Vestidos Formal', 'Vestidos Playa'],
            'season_weights': {
                'Winter': 0.05,
                'Fall': 0.10,
                'Spring': 0.35,
                'Summer': 0.50
            }
        },
        'Activewear': {
            'items': ['Tops Deportivos', 'Leggings', 'Shorts Deportivos'],
            'season_weights': {
                'Winter': 0.15,
                'Fall': 0.20,
                'Spring': 0.30,
                'Summer': 0.35
            }
        },
        'Swimwear': {
            'items': ['Trajes de Baño', 'Bikinis', 'Pareos'],
            'season_weights': {
                'Winter': 0.00,
                'Fall': 0.00,
                'Spring': 0.15,
                'Summer': 0.85
            }
        },
        'Accessories': {
            'items': ['Gorros', 'Bufandas', 'Guantes', 'Sombreros'],
            'season_weights': {
                'Winter': 0.70,
                'Fall': 0.20,
                'Spring': 0.08,
                'Summer': 0.02
            }
        }
    }
    
    # Distribución más realista por temporada
    season_distribution = {
        'Winter': int(n_samples * 0.35),
        'Summer': int(n_samples * 0.30),
        'Fall': int(n_samples * 0.20),
        'Spring': int(n_samples * 0.15)
    }
    
    seasons = []
    for season, count in season_distribution.items():
        seasons.extend([season] * count)
    
    while len(seasons) < n_samples:
        seasons.append(np.random.choice(list(season_distribution.keys())))
    
    np.random.shuffle(seasons)
    
    # Generar categoría y subcategoría con mayor coherencia
    categories = []
    subcategories = []
    
    for season in seasons:
        category_probs = {
            cat: info['season_weights'][season]
            for cat, info in product_categories.items()
        }
        
        max_prob = max(category_probs.values())
        for k in category_probs:
            if category_probs[k] == max_prob:
                category_probs[k] *= 1.5
        
        total = sum(category_probs.values())
        category_probs = {k: v/total for k, v in category_probs.items()}
        
        cat = np.random.choice(
            list(category_probs.keys()),
            p=list(category_probs.values())
        )
        categories.append(cat)
        
        if season in ['Winter', 'Fall']:
            weights = [3.0 if item in ['Abrigos', 'Chaquetas', 'Pantalones'] else 1.0 
                      for item in product_categories[cat]['items']]
        else:
            weights = [3.0 if item in ['Shorts', 'Vestidos Playa', 'Tops Deportivos'] else 1.0 
                      for item in product_categories[cat]['items']]
        
        weights = [w/sum(weights) for w in weights]
        
        subcat = np.random.choice(
            product_categories[cat]['items'],
            p=weights
        )
        subcategories.append(subcat)
    
    # 2. Materiales con mayor coherencia estacional
    materials = {
        'Natural': {
            'items': ['Algodón', 'Lana', 'Lino', 'Seda'],
            'season_weights': {
                'Winter': {'Lana': 0.85, 'Algodón': 0.10, 'Seda': 0.05, 'Lino': 0.00},
                'Summer': {'Lino': 0.60, 'Algodón': 0.35, 'Seda': 0.05, 'Lana': 0.00},
                'Spring': {'Algodón': 0.50, 'Lino': 0.30, 'Seda': 0.20, 'Lana': 0.00},
                'Fall': {'Lana': 0.60, 'Algodón': 0.25, 'Seda': 0.10, 'Lino': 0.05}
            }
        },
        'Sintético': {
            'items': ['Poliéster', 'Nylon', 'Spandex'],
            'season_props': {'Winter': 0.45, 'Summer': 0.25, 'Spring': 0.15, 'Fall': 0.15}
        },
        'Técnico': {
            'items': ['Gore-Tex', 'Dri-FIT', 'Thinsulate'],
            'season_props': {'Winter': 0.70, 'Summer': 0.05, 'Spring': 0.10, 'Fall': 0.15}
        }
    }
    
    material_categories = []
    material_types = []
    
    for cat, season in zip(categories, seasons):
        if cat in ['Outerwear', 'Activewear']:
            if season in ['Winter', 'Fall']:
                mat_weights = {'Sintético': 0.15, 'Técnico': 0.75, 'Natural': 0.10}
            else:
                mat_weights = {'Sintético': 0.40, 'Técnico': 0.45, 'Natural': 0.15}
        elif cat == 'Swimwear':
            mat_weights = {'Sintético': 0.95, 'Técnico': 0.05, 'Natural': 0.00}
        else:
            if season in ['Summer', 'Spring']:
                mat_weights = {'Natural': 0.80, 'Sintético': 0.15, 'Técnico': 0.05}
            else:
                mat_weights = {'Natural': 0.60, 'Sintético': 0.25, 'Técnico': 0.15}
        
        mat_cat = np.random.choice(
            list(mat_weights.keys()),
            p=list(mat_weights.values())
        )
        material_categories.append(mat_cat)
        
        if mat_cat == 'Natural':
            weights = materials['Natural']['season_weights'][season]
        elif mat_cat == 'Técnico':
            if season == 'Winter':
                weights = {'Gore-Tex': 0.35, 'Thinsulate': 0.60, 'Dri-FIT': 0.05}
            else:
                weights = {'Dri-FIT': 0.70, 'Gore-Tex': 0.25, 'Thinsulate': 0.05}
        else:
            if cat == 'Swimwear':
                weights = {'Nylon': 0.60, 'Spandex': 0.35, 'Poliéster': 0.05}
            else:
                weights = {'Poliéster': 0.50, 'Nylon': 0.25, 'Spandex': 0.25}
        
        mat_type = np.random.choice(
            list(weights.keys()),
            p=list(weights.values())
        )
        material_types.append(mat_type)
    
    # 3. Características técnicas con mayor coherencia
    thickness = []
    waterproof = []
    thermal_rating = []
    
    for cat, season, mat in zip(categories, seasons, material_types):
        if season == 'Winter':
            if cat in ['Outerwear', 'Accessories']:
                thick_weights = {'Grueso': 0.85, 'Medio': 0.15, 'Ligero': 0.00}
            else:
                thick_weights = {'Grueso': 0.60, 'Medio': 0.35, 'Ligero': 0.05}
        elif season == 'Summer':
            thick_weights = {'Ligero': 0.85, 'Medio': 0.15, 'Grueso': 0.00}
        else:
            thick_weights = {'Medio': 0.70, 'Ligero': 0.20, 'Grueso': 0.10}
        
        thickness.append(np.random.choice(
            list(thick_weights.keys()),
            p=list(thick_weights.values())
        ))
        
        if cat == 'Outerwear' or mat == 'Gore-Tex':
            if season in ['Winter', 'Fall']:
                water_weights = {'Impermeable': 0.85, 'Repelente': 0.15, 'No': 0.00}
            else:
                water_weights = {'Impermeable': 0.60, 'Repelente': 0.35, 'No': 0.05}
        elif cat == 'Swimwear':
            water_weights = {'Repelente': 0.95, 'No': 0.05, 'Impermeable': 0.00}
        else:
            water_weights = {'No': 0.80, 'Repelente': 0.15, 'Impermeable': 0.05}
        
        waterproof.append(np.random.choice(
            list(water_weights.keys()),
            p=list(water_weights.values())
        ))
        
        if season == 'Winter':
            if cat in ['Outerwear', 'Accessories']:
                thermal_weights = {'Muy Alto': 0.70, 'Alto': 0.25, 'Medio': 0.05}
            else:
                thermal_weights = {'Alto': 0.60, 'Medio': 0.35, 'Bajo': 0.05}
        elif season == 'Summer':
            thermal_weights = {'Bajo': 0.95, 'Medio': 0.05}
        else:
            thermal_weights = {'Medio': 0.70, 'Bajo': 0.20, 'Alto': 0.10}
        
        thermal_rating.append(np.random.choice(
            list(thermal_weights.keys()),
            p=list(thermal_weights.values())
        ))
    
    # 4. Características de diseño más distintivas
    color_families = []
    patterns = []
    styles = []
    
    style_mapping = {
        ('Winter', 'Outerwear'): {'Formal': 0.50, 'Casual': 0.30, 'Deportivo': 0.20},
        ('Summer', 'Swimwear'): {'Playa': 0.70, 'Deportivo': 0.20, 'Casual': 0.10},
        ('Spring', 'Dresses'): {'Casual': 0.40, 'Formal': 0.30, 'Playa': 0.30}
    }
    
    for season, cat in zip(seasons, categories):
        if season == 'Summer':
            color_weights = {'Vibrante': 0.55, 'Pastel': 0.30, 'Neutro': 0.15, 'Oscuro': 0.00}
        elif season == 'Winter':
            color_weights = {'Oscuro': 0.55, 'Neutro': 0.35, 'Vibrante': 0.07, 'Pastel': 0.03}
        elif season == 'Spring':
            color_weights = {'Pastel': 0.45, 'Vibrante': 0.35, 'Neutro': 0.15, 'Oscuro': 0.05}
        else:
            color_weights = {'Neutro': 0.45, 'Oscuro': 0.35, 'Vibrante': 0.15, 'Pastel': 0.05}
        
        color_families.append(np.random.choice(
            list(color_weights.keys()),
            p=list(color_weights.values())
        ))
        
        if season == 'Summer':
            pattern_weights = {'Floral': 0.45, 'Estampado': 0.30, 'Sólido': 0.15, 'Rayas': 0.10}
        elif season == 'Winter':
            pattern_weights = {'Sólido': 0.45, 'Cuadros': 0.35, 'Rayas': 0.15, 'Estampado': 0.05}
        elif season == 'Spring':
            pattern_weights = {'Floral': 0.35, 'Estampado': 0.30, 'Sólido': 0.20, 'Rayas': 0.15}
        else:  # Fall
            pattern_weights = {'Sólido': 0.35, 'Cuadros': 0.35, 'Rayas': 0.20, 'Estampado': 0.10}
        
        patterns.append(np.random.choice(
            list(pattern_weights.keys()),
            p=list(pattern_weights.values())
        ))
        
        # Estilo con mapeo específico por temporada y categoría
        style_key = (season, cat)
        if style_key in style_mapping:
            weights = style_mapping[style_key]
        else:
            if season == 'Summer':
                weights = {'Casual': 0.30, 'Deportivo': 0.30, 'Formal': 0.10, 'Playa': 0.30}
            elif season == 'Winter':
                weights = {'Formal': 0.45, 'Casual': 0.35, 'Deportivo': 0.15, 'Playa': 0.05}
            elif season == 'Spring':
                weights = {'Casual': 0.40, 'Deportivo': 0.25, 'Formal': 0.25, 'Playa': 0.10}
            else:  # Fall
                weights = {'Casual': 0.35, 'Deportivo': 0.25, 'Formal': 0.35, 'Playa': 0.05}
        
        styles.append(np.random.choice(
            list(weights.keys()),
            p=list(weights.values())
        ))

    # Crear DataFrame
    df = pd.DataFrame({
        'Product_Category': categories,
        'Subcategory': subcategories,
        'Material_Category': material_categories,
        'Material_Type': material_types,
        'Thickness': thickness,
        'Waterproof_Rating': waterproof,
        'Thermal_Rating': thermal_rating,
        'Color_Family': color_families,
        'Pattern': patterns,
        'Style': styles,
        'Season': seasons
    })
    
    # Validación y corrección de combinaciones inválidas
    invalid_combinations = (
        ((df['Season'] == 'Summer') & (df['Thermal_Rating'].isin(['Muy Alto', 'Alto']))) |
        ((df['Season'] == 'Winter') & (df['Product_Category'] == 'Swimwear')) |
        ((df['Season'] == 'Summer') & (df['Thickness'] == 'Grueso')) |
        ((df['Season'] == 'Winter') & (df['Material_Type'] == 'Lino')) |
        ((df['Season'] == 'Summer') & (df['Material_Type'] == 'Lana'))
    )
    
    # Corregir combinaciones inválidas
    df.loc[invalid_combinations, ['Season', 'Thermal_Rating', 'Thickness']] = \
        df.loc[invalid_combinations].apply(corregir_combinacion, axis=1)
    
    # Calcular y mostrar estadísticas
    print("\n=== Dataset Generado Exitosamente ===")
    print(f"\nNúmero total de registros: {len(df)}")
    
    print("\nDistribución de Temporadas:")
    print(df['Season'].value_counts(normalize=True))
    
    print("\nDistribución por Categoría de Producto y Temporada:")
    print(pd.crosstab(df['Product_Category'], df['Season'], normalize='index'))
    
    print("\nFuerza de asociación con Season (V de Cramér):")
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Season':
            cramers_v = calculate_cramers_v(df[column], df['Season'])
            print(f"{column}: {cramers_v:.3f}")
    
    # Guardar dataset
    df.to_csv('demanda_producto.csv', index=False, sep=';')
    
    return df

# Generar datos
if __name__ == "__main__":
    df = generar_dataset_completo()