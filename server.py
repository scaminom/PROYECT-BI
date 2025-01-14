from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os

app = Flask(__name__)

# Rutas de los archivos modelo y preprocessors
MODEL_PATH = 'modelos/modelo.h5'
PREPROCESSORS_PATH = 'datos/preprocessors.pkl'

# Cargar modelo y preprocessors
model = load_model(MODEL_PATH)

# Cargar los preprocessors
with open(PREPROCESSORS_PATH, 'rb') as f:
    preprocessors = pickle.load(f)

# Obtener los encoders y scaler
label_encoders = preprocessors['label_encoders']
label_encoder_y = preprocessors['label_encoder_y']
scaler = preprocessors['scaler']

# Definir el orden exacto de las características
FEATURE_NAMES = [
    'Product_Category', 'Subcategory', 'Material_Category', 
    'Material_Type', 'Thickness', 'Waterproof_Rating', 
    'Thermal_Rating', 'Color_Family', 'Pattern', 'Style'
]

# Definir las relaciones entre categorías y subcategorías según el dataset generado
PRODUCT_RELATIONS = {
    'Outerwear': ['Chaquetas', 'Abrigos', 'Impermeables', 'Chalecos'],
    'Tops': ['Camisetas', 'Blusas', 'Polos', 'Sudaderas'],
    'Bottoms': ['Pantalones', 'Shorts', 'Faldas', 'Jeans'],
    'Dresses': ['Vestidos Casual', 'Vestidos Formal', 'Vestidos Playa'],
    'Activewear': ['Tops Deportivos', 'Leggings', 'Shorts Deportivos'],
    'Swimwear': ['Trajes de Baño', 'Bikinis', 'Pareos'],
    'Accessories': ['Gorros', 'Bufandas', 'Guantes', 'Sombreros']
}

MATERIAL_RELATIONS = {
    'Natural': ['Algodón', 'Lana', 'Lino', 'Seda'],
    'Sintético': ['Poliéster', 'Nylon', 'Spandex'],
    'Técnico': ['Gore-Tex', 'Dri-FIT', 'Thinsulate']
}

# Definir relaciones de estilos según la temporada
STYLE_SEASON_RELATIONS = {
    'Summer': {'Casual': 0.30, 'Deportivo': 0.30, 'Formal': 0.10, 'Playa': 0.30},
    'Winter': {'Formal': 0.45, 'Casual': 0.35, 'Deportivo': 0.15, 'Playa': 0.05},
    'Spring': {'Casual': 0.40, 'Deportivo': 0.25, 'Formal': 0.25, 'Playa': 0.10},
    'Fall': {'Casual': 0.35, 'Deportivo': 0.25, 'Formal': 0.35, 'Playa': 0.05}
}

@app.route('/')
def home():
    """Página de inicio con formulario de predicción"""
    categories = {
        'product_categories': list(PRODUCT_RELATIONS.keys()),
        'material_categories': list(MATERIAL_RELATIONS.keys()),
        'thicknesses': label_encoders['Thickness'].classes_.tolist(),
        'waterproof_ratings': label_encoders['Waterproof_Rating'].classes_.tolist(),
        'thermal_ratings': label_encoders['Thermal_Rating'].classes_.tolist(),
        'color_families': label_encoders['Color_Family'].classes_.tolist(),
        'patterns': label_encoders['Pattern'].classes_.tolist(),
        'styles': label_encoders['Style'].classes_.tolist()
    }
    return render_template('index.html', categories=categories)

@app.route('/get_subcategories/<category>')
def get_subcategories(category):
    """Endpoint para obtener subcategorías basadas en la categoría"""
    try:
        subcategories = PRODUCT_RELATIONS.get(category, [])
        return jsonify({
            'success': True, 
            'subcategories': subcategories
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/get_material_types/<category>')
def get_material_types(category):
    """Endpoint para obtener tipos de materiales basados en la categoría"""
    try:
        material_types = MATERIAL_RELATIONS.get(category, [])
        return jsonify({
            'success': True, 
            'material_types': material_types
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/get_styles/<season>')
def get_styles(season):
    """Endpoint para obtener estilos recomendados según la temporada"""
    try:
        styles = STYLE_SEASON_RELATIONS.get(season, {})
        return jsonify({
            'success': True, 
            'styles': styles
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predecir la temporada de un producto"""
    try:
        data = request.get_json()
        
        # Validar combinaciones según las reglas del negocio
        validation_result = validate_combinations(data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['message']
            })
        
        # Procesar y predecir
        features = process_input(data)
        prediction = model.predict(features)
        
        # Convertir predicciones a porcentajes
        probabilities = {}
        for i, class_name in enumerate(label_encoder_y.classes_):
            probabilities[class_name] = float(prediction[0][i] * 100)
        
        return jsonify({
            'success': True,
            'predictions': probabilities,
            'recommended_season': max(probabilities, key=probabilities.get)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def validate_combinations(data):
    """Validar combinaciones de características según reglas del negocio"""
    invalid_combinations = [
        {
            'condition': data['product_category'] == 'Swimwear' and data['material_type'] == 'Lana',
            'message': 'Los trajes de baño no pueden ser de lana'
        },
        {
            'condition': data['product_category'] == 'Outerwear' and data['thickness'] == 'Ligero' and data['thermal_rating'] == 'Muy Alto',
            'message': 'Una prenda exterior ligera no puede tener rating térmico muy alto'
        },
        {
            'condition': data['material_type'] == 'Gore-Tex' and data['waterproof_rating'] == 'No',
            'message': 'Las prendas Gore-Tex deben ser impermeables o repelentes'
        }
    ]
    
    for combo in invalid_combinations:
        if combo['condition']:
            return {
                'valid': False,
                'message': combo['message']
            }
    
    return {'valid': True}

def process_input(data):
    """Procesa los datos de entrada usando los encoders"""
    processed_values = {}
    for feature in FEATURE_NAMES:
        key = feature.lower()
        processed_values[feature] = label_encoders[feature].transform([data[key]])[0]
    
    input_df = pd.DataFrame([processed_values], columns=FEATURE_NAMES)
    return input_df.values

if __name__ == '__main__':
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('datos', exist_ok=True)
    app.run(debug=True)