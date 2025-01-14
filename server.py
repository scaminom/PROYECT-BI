from flask import Flask, request, jsonify, render_template
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Cargar el modelo y los preprocessors
MODEL_PATH = 'modelos/modelo.h5'
PREPROCESSORS_PATH = 'datos/preprocessors.pkl'
model = load_model(MODEL_PATH)

# Cargar los preprocessors
with open(PREPROCESSORS_PATH, 'rb') as f:
    preprocessors = pickle.load(f)

# Obtener los encoders y scaler
label_encoders = preprocessors['label_encoders']
label_encoder_y = preprocessors['label_encoder_y']
scaler = preprocessors['scaler']

# Definir el orden exacto de las características usadas en el entrenamiento
FEATURE_NAMES = [
    'Lead Time Days',
    'Typical Weight Per Unit',
    'Product_Base_Price',
    'Buying Group',
    'Customer_Category',
    'Sales Territory',
    'Continent',
    'Country',
    'Month',
    'Fiscal Year',
    'Is Holiday',
    'Quantity',
    'Package'
]

@app.route('/')
def home():
    # Obtener las categorías únicas para cada campo
    categories = {
        'buying_groups': label_encoders['Buying Group'].classes_.tolist(),
        'customer_categories': label_encoders['Customer_Category'].classes_.tolist(),
        'sales_territories': label_encoders['Sales Territory'].classes_.tolist(),
        'continents': label_encoders['Continent'].classes_.tolist(),
        'countries': label_encoders['Country'].classes_.tolist(),
        'packages': label_encoders['Package'].classes_.tolist()
    }
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        data = request.get_json()
        
        # Preparar los datos para la predicción
        features = process_input(data)
        
        # Realizar predicción
        prediction = model.predict(features)
        
        # Convertir predicciones a porcentajes y usar las clases correctas
        probabilities = {}
        for i, class_name in enumerate(label_encoder_y.classes_):
            probabilities[class_name] = float(prediction[0][i] * 100)
        
        return jsonify({
            'success': True,
            'predictions': probabilities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def process_input(data):
    """
    Procesa los datos de entrada usando los encoders y scaler correctos
    """
    # Crear un diccionario con los valores procesados
    processed_values = {
        'Lead Time Days': float(data['lead_time']),
        'Typical Weight Per Unit': float(data['weight']),
        'Product_Base_Price': float(data['base_price']),
        'Buying Group': label_encoders['Buying Group'].transform([data['buying_group']])[0],
        'Customer_Category': label_encoders['Customer_Category'].transform([data['customer_category']])[0],
        'Sales Territory': label_encoders['Sales Territory'].transform([data['sales_territory']])[0],
        'Continent': label_encoders['Continent'].transform([data['continent']])[0],
        'Country': label_encoders['Country'].transform([data['country']])[0],
        'Month': int(data['month']),
        'Fiscal Year': int(data['fiscal_year']),
        'Is Holiday': int(data['is_holiday']),
        'Quantity': float(data['quantity']),
        'Package': label_encoders['Package'].transform([data['package']])[0]
    }
    
    # Crear DataFrame con los nombres de características correctos y en el orden correcto
    input_df = pd.DataFrame([processed_values], columns=FEATURE_NAMES)
    
    # Aplicar el scaler solo a las columnas numéricas
    columnas_numericas = ['Lead Time Days', 'Typical Weight Per Unit', 'Product_Base_Price',
                         'Month', 'Fiscal Year', 'Quantity']
    
    input_df[columnas_numericas] = scaler.transform(input_df[columnas_numericas])
    
    return input_df.values

if __name__ == '__main__':
    app.run(debug=True)
