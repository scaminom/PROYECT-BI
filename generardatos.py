
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import chi2_contingency

def generar_dataset_completo(n_samples=95060, random_state=42):
    """
    Genera un dataset completo con features más relevantes y mejor separación entre clases.
    """
    np.random.seed(random_state)
    
    # 1. Características base del producto con pesos más diferenciados por temporada
    product_categories = {
        'Outerwear': {
            'items': ['Chaquetas', 'Abrigos', 'Impermeables', 'Chalecos'],
            'season_weights': {
                'Winter': 0.8,
                'Fall': 0.15,
                'Spring': 0.05,
                'Summer': 0.0
            }
        },
        'Tops': {
            'items': ['Camisetas', 'Blusas', 'Polos', 'Sudaderas'],
            'season_weights': {
                'Winter': 0.1,
                'Fall': 0.2,
                'Spring': 0.3,
                'Summer': 0.4
            }
        },
        'Bottoms': {
            'items': ['Pantalones', 'Shorts', 'Faldas', 'Jeans'],
            'season_weights': {
                'Winter': 0.4,
                'Fall': 0.3,
                'Spring': 0.2,
                'Summer': 0.1
            }
        },
        'Dresses': {
            'items': ['Vestidos Casual', 'Vestidos Formal', 'Vestidos Playa'],
            'season_weights': {
                'Winter': 0.05,
                'Fall': 0.15,
                'Spring': 0.3,
                'Summer': 0.5
            }
        },
        'Activewear': {
            'items': ['Tops Deportivos', 'Leggings', 'Shorts Deportivos'],
            'season_weights': {
                'Winter': 0.1,
                'Fall': 0.2,
                'Spring': 0.3,
                'Summer': 0.4
            }
        },
        'Swimwear': {
            'items': ['Trajes de Baño', 'Bikinis', 'Pareos'],
            'season_weights': {
                'Winter': 0.0,
                'Fall': 0.0,
                'Spring': 0.2,
                'Summer': 0.8
            }
        },
        'Accessories': {
            'items': ['Gorros', 'Bufandas', 'Guantes', 'Sombreros'],
            'season_weights': {
                'Winter': 0.6,
                'Fall': 0.25,
                'Spring': 0.1,
                'Summer': 0.05
            }
        }
    }
    
    # Asignar temporadas con distribución controlada
    season_distribution = {
        'Winter': int(n_samples * 0.25),
        'Spring': int(n_samples * 0.25),
        'Summer': int(n_samples * 0.25),
        'Fall': int(n_samples * 0.25)
    }
    
    seasons = []
    for season, count in season_distribution.items():
        seasons.extend([season] * count)
    
    # Ajustar por si hay diferencia debido a redondeo
    while len(seasons) < n_samples:
        seasons.append(np.random.choice(list(season_distribution.keys())))
    
    np.random.shuffle(seasons)
    
    # Generar categoría y subcategoría con mayor coherencia
    categories = []
    subcategories = []
    
    for season in seasons:
        # Ajustar probabilidades según temporada
        category_probs = {
            cat: info['season_weights'][season]
            for cat, info in product_categories.items()
        }
        
        # Fortalecer las probabilidades dominantes
        max_prob = max(category_probs.values())
        for k in category_probs:
            if category_probs[k] == max_prob:
                category_probs[k] *= 1.2  # Aumentar probabilidad dominante
        
        # Normalizar probabilidades
        total = sum(category_probs.values())
        category_probs = {k: v/total for k, v in category_probs.items()}
        
        # Seleccionar categoría
        cat = np.random.choice(
            list(category_probs.keys()),
            p=list(category_probs.values())
        )
        categories.append(cat)
        
        # Seleccionar subcategoría con coherencia estacional
        if season in ['Winter', 'Fall']:
            weights = [2.0 if item in ['Abrigos', 'Chaquetas', 'Pantalones'] else 1.0 
                      for item in product_categories[cat]['items']]
        else:
            weights = [2.0 if item in ['Shorts', 'Vestidos Playa', 'Tops Deportivos'] else 1.0 
                      for item in product_categories[cat]['items']]
        
        # Normalizar pesos
        weights = [w/sum(weights) for w in weights]
        
        subcat = np.random.choice(
            product_categories[cat]['items'],
            p=weights
        )
        subcategories.append(subcat)
    
    # 2. Características del material con mayor diferenciación estacional
    materials = {
        'Natural': {
            'items': ['Algodón', 'Lana', 'Lino', 'Seda'],
            'season_weights': {
                'Winter': {'Lana': 0.7, 'Algodón': 0.2, 'Seda': 0.1, 'Lino': 0.0},
                'Summer': {'Lino': 0.5, 'Algodón': 0.4, 'Seda': 0.1, 'Lana': 0.0},
                'Spring': {'Algodón': 0.5, 'Lino': 0.3, 'Seda': 0.2, 'Lana': 0.0},
                'Fall': {'Lana': 0.5, 'Algodón': 0.3, 'Seda': 0.1, 'Lino': 0.1}
            }
        },
        'Sintético': {
            'items': ['Poliéster', 'Nylon', 'Spandex'],
            'season_props': {'Winter': 0.4, 'Summer': 0.3, 'Spring': 0.2, 'Fall': 0.1}
        },
        'Técnico': {
            'items': ['Gore-Tex', 'Dri-FIT', 'Thinsulate'],
            'season_props': {'Winter': 0.6, 'Summer': 0.1, 'Spring': 0.1, 'Fall': 0.2}
        }
    }
    
    material_categories = []
    material_types = []
    
    for cat, season in zip(categories, seasons):
        # Ajustar pesos según categoría y temporada
        if cat in ['Outerwear', 'Activewear']:
            if season in ['Winter', 'Fall']:
                mat_weights = {'Sintético': 0.2, 'Técnico': 0.6, 'Natural': 0.2}
            else:
                mat_weights = {'Sintético': 0.4, 'Técnico': 0.4, 'Natural': 0.2}
        elif cat == 'Swimwear':
            mat_weights = {'Sintético': 0.9, 'Técnico': 0.1, 'Natural': 0.0}
        else:
            if season in ['Summer', 'Spring']:
                mat_weights = {'Natural': 0.7, 'Sintético': 0.2, 'Técnico': 0.1}
            else:
                mat_weights = {'Natural': 0.5, 'Sintético': 0.3, 'Técnico': 0.2}
        
        mat_cat = np.random.choice(
            list(mat_weights.keys()),
            p=list(mat_weights.values())
        )
        material_categories.append(mat_cat)
        
        # Selección de material específico con mayor coherencia
        if mat_cat == 'Natural':
            weights = materials['Natural']['season_weights'][season]
        elif mat_cat == 'Técnico':
            if season == 'Winter':
                weights = {'Gore-Tex': 0.4, 'Thinsulate': 0.5, 'Dri-FIT': 0.1}
            else:
                weights = {'Dri-FIT': 0.6, 'Gore-Tex': 0.3, 'Thinsulate': 0.1}
        else:
            if cat == 'Swimwear':
                weights = {'Nylon': 0.5, 'Spandex': 0.4, 'Poliéster': 0.1}
            else:
                weights = {'Poliéster': 0.4, 'Nylon': 0.3, 'Spandex': 0.3}
        
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
        # Grosor con reglas más estrictas
        if season == 'Winter':
            if cat in ['Outerwear', 'Accessories']:
                thick_weights = {'Grueso': 0.8, 'Medio': 0.2, 'Ligero': 0.0}
            else:
                thick_weights = {'Grueso': 0.5, 'Medio': 0.4, 'Ligero': 0.1}
        elif season == 'Summer':
            thick_weights = {'Ligero': 0.8, 'Medio': 0.2, 'Grueso': 0.0}
        else:
            thick_weights = {'Medio': 0.6, 'Ligero': 0.3, 'Grueso': 0.1}
        
        thickness.append(np.random.choice(
            list(thick_weights.keys()),
            p=list(thick_weights.values())
        ))
        
        # Impermeabilidad con reglas más específicas
        if cat == 'Outerwear' or mat == 'Gore-Tex':
            if season in ['Winter', 'Fall']:
                water_weights = {'Impermeable': 0.8, 'Repelente': 0.2, 'No': 0.0}
            else:
                water_weights = {'Impermeable': 0.5, 'Repelente': 0.4, 'No': 0.1}
        elif cat == 'Swimwear':
            water_weights = {'Repelente': 0.9, 'No': 0.1, 'Impermeable': 0.0}
        else:
            water_weights = {'No': 0.7, 'Repelente': 0.2, 'Impermeable': 0.1}
        
        waterproof.append(np.random.choice(
            list(water_weights.keys()),
            p=list(water_weights.values())
        ))
        
        # Rating térmico con mayor coherencia
        if season == 'Winter':
            if cat in ['Outerwear', 'Accessories']:
                thermal_weights = {'Muy Alto': 0.6, 'Alto': 0.3, 'Medio': 0.1}
            else:
                thermal_weights = {'Alto': 0.5, 'Medio': 0.4, 'Bajo': 0.1}
        elif season == 'Summer':
            thermal_weights = {'Bajo': 0.9, 'Medio': 0.1}
        else:
            thermal_weights = {'Medio': 0.6, 'Bajo': 0.3, 'Alto': 0.1}
        
        thermal_rating.append(np.random.choice(
            list(thermal_weights.keys()),
            p=list(thermal_weights.values())
        ))
    
    # 4. Características de diseño más distintivas por temporada
    color_families = []
    patterns = []
    styles = []
    
    for season in seasons:
        # Color con mayor diferenciación estacional
        if season == 'Summer':
            color_weights = {'Vibrante': 0.5, 'Pastel': 0.3, 'Neutro': 0.2, 'Oscuro': 0.0}
        elif season == 'Winter':
            color_weights = {'Oscuro': 0.5, 'Neutro': 0.3, 'Vibrante': 0.1, 'Pastel': 0.1}
        elif season == 'Spring':
            color_weights = {'Pastel': 0.4, 'Vibrante': 0.3, 'Neutro': 0.2, 'Oscuro': 0.1}
        else:  # Fall
            color_weights = {'Neutro': 0.4, 'Oscuro': 0.3, 'Vibrante': 0.2, 'Pastel': 0.1}
        
        color_families.append(np.random.choice(
            list(color_weights.keys()),
            p=list(color_weights.values())
        ))
        
        # Patrón con mayor coherencia estacional
        if season == 'Summer':
            pattern_weights = {'Floral': 0.4, 'Estampado': 0.3, 'Sólido': 0.2, 'Rayas': 0.1}
        elif season == 'Winter':
            pattern_weights = {'Sólido': 0.4, 'Cuadros': 0.3, 'Rayas': 0.2, 'Estampado': 0.1}
        elif season == 'Spring':
            pattern_weights = {'Floral': 0.3, 'Estampado': 0.3, 'Sólido': 0.2, 'Rayas': 0.2}
        else:  # Fall
            pattern_weights = {'Sólido': 0.3, 'Cuadros': 0.3, 'Rayas': 0.2, 'Estampado': 0.2}
        
        patterns.append(np.random.choice(
            list(pattern_weights.keys()),
            p=list(pattern_weights.values())
        ))
        # Estilo

        if season == 'Summer':
            style_weights = {'Casual': 0.2, 'Deportivo': 0.3, 'Formal': 0.1, 'Playa': 0.4}
        elif season == 'Winter':
            style_weights = {'Formal': 0.4, 'Casual': 0.3, 'Deportivo': 0.2, 'Playa': 0.1}
        elif season == 'Spring':
            style_weights = {'Casual': 0.4, 'Deportivo': 0.2, 'Formal': 0.3, 'Playa': 0.1}
        else:  # Fall
            style_weights = {'Casual': 0.2, 'Deportivo': 0.3, 'Formal': 0.3, 'Playa': 0.2}
        
        styles.append(np.random.choice(
            list(style_weights.keys()),
            p=list(style_weights.values())
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
    
    # Validación de coherencia
    invalid_combinations = (
        ((df['Season'] == 'Summer') & (df['Thermal_Rating'] == 'Muy Alto')) |
        ((df['Season'] == 'Winter') & (df['Product_Category'] == 'Swimwear')) |
        ((df['Season'] == 'Summer') & (df['Thickness'] == 'Grueso') & 
         (df['Product_Category'].isin(['Tops', 'Dresses'])))
    )
    
    # Corregir combinaciones inválidas
    df.loc[invalid_combinations, 'Season'] = df.loc[invalid_combinations].apply(
        lambda row: np.random.choice(['Spring', 'Fall']), axis=1
    )
    
    # Guardar dataset
    df.to_csv('demanda_producto.csv', index=False, sep=';')
    
    # Imprimir estadísticas
    print("\n=== Dataset Generado Exitosamente ===")
    print(f"\nNúmero total de registros: {len(df)}")
    
    print("\nDistribución de Temporadas:")
    print(df['Season'].value_counts(normalize=True))
    
    print("\nDistribución por Categoría de Producto y Temporada:")
    print(pd.crosstab(df['Product_Category'], df['Season'], normalize='index'))
    
    # Calcular y mostrar el V de Cramér para cada variable
    print("\nFuerza de asociación con Season (V de Cramér):")
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'Season':
            cramers_v = calculate_cramers_v(df[column], df['Season'])
            print(f"{column}: {cramers_v:.3f}")
    
    return df

def calculate_cramers_v(x, y):
    """
    Calcula el coeficiente V de Cramér para variables categóricas
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]  # TODO: Implementar cálculo real
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# Generar datos
if __name__ == "__main__":
    df = generar_dataset_completo()