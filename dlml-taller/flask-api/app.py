from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Cargar el modelo y preprocesadores
model = joblib.load('model_arrest.pkl')
scaler = joblib.load('scaler.pkl')
ord_enc = joblib.load('ordinal_encoder.pkl')

# Definir las columnas requeridas (excluyendo las eliminadas en el entrenamiento)
required_columns = [
    'PRIMARY DESCRIPTION', 'SECONDARY DESCRIPTION', 'LOCATION DESCRIPTION',
    'FBI CD', 'IUCR', 'BEAT', 'WARD', 'X COORDINATE'
]

categorical_cols = ['PRIMARY DESCRIPTION', 'SECONDARY DESCRIPTION', 'LOCATION DESCRIPTION', 'FBI CD', 'IUCR']
numeric_cols = ['BEAT', 'WARD', 'X COORDINATE']

def preprocess_data(df):
    # Asegurar que todas las columnas requeridas estén presentes
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan  # Asignar NaN si falta la columna

    # Llenar valores faltantes en categóricas y numéricas
    df[categorical_cols] = df[categorical_cols].astype(str).fillna('UNKNOWN')
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(0)  # O usar la media si es apropiado

    # Transformar variables categóricas con OrdinalEncoder
    df[categorical_cols] = ord_enc.transform(df[categorical_cols])

    # Escalar variables numéricas
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Asegurar que las columnas estén en el orden correcto
    df = df[categorical_cols + numeric_cols]

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON de la solicitud
        data = request.get_json()
        df = pd.DataFrame([data])

        # Preprocesar los datos
        df_preprocessed = preprocess_data(df)

        # Realizar la predicción
        prediction = model.predict(df_preprocessed)

        # Devolver la predicción como JSON
        return jsonify({'prediction': 'Y' if prediction[0] == 1 else 'N'})
    except Exception as e:
        # Manejar errores y devolver mensaje al cliente
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
