import os
from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Usar un backend sin GUI para Matplotlib
import matplotlib.pyplot as plt

# Crear la aplicación Flask
app = Flask(__name__)

# Obtener el directorio base del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas a los modelos y codificadores
model_path = os.path.join(BASE_DIR, '../models/modelo_time_occ.pkl')
encoder_area_path = os.path.join(BASE_DIR, 'enconders/encoder_area.pkl')
encoder_crm_path = os.path.join(BASE_DIR, 'enconders/encoder_crm.pkl')
encoder_premis_path = os.path.join(BASE_DIR, 'enconders/encoder_premis.pkl')
encoder_day_night_path = os.path.join(BASE_DIR, 'enconders/encoder_day_night.pkl')

# Cargar el modelo y los codificadores
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_area_path, 'rb') as f:
    encoder_area = pickle.load(f)

with open(encoder_crm_path, 'rb') as f:
    encoder_crm = pickle.load(f)

with open(encoder_premis_path, 'rb') as f:
    encoder_premis = pickle.load(f)

with open(encoder_day_night_path, 'rb') as f:
    encoder_day_night = pickle.load(f)

def generate_pdf(prediction, time_of_day, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Reporte de Predicción de Crímenes", ln=True, align='C')

    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, txt=f"Predicción de Hora: {prediction}", ln=True)
    pdf.cell(0, 10, txt=f"Rango Horario: {time_of_day}", ln=True)

    plt.figure(figsize=(6, 4))
    plt.bar(['Día', 'Noche'], [60, 40], color='blue', alpha=0.7)
    plt.title('Frecuencia Estimada por Rango Horario')
    plt.xlabel('Rango Horario')
    plt.ylabel('Frecuencia')
    chart_path = os.path.join(BASE_DIR, 'frequency_chart.png')
    plt.savefig(chart_path)
    plt.close()

    pdf.image(chart_path, x=10, y=50, w=180)
    pdf.output(pdf_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        area_name = data['area_name']
        crm_cd_desc = data['crm_cd_desc']
        premis_desc = data['premis_desc']
        day_night = data['day_night']

        area_encoded = encoder_area.transform([area_name])[0]
        crm_encoded = encoder_crm.transform([crm_cd_desc])[0]
        premis_encoded = encoder_premis.transform([premis_desc])[0]
        day_night_encoded = encoder_day_night.transform([day_night])[0]

        features = np.array([[area_encoded, crm_encoded, premis_encoded, day_night_encoded]])

        predicted_time = model.predict(features)[0]
        hours = int(predicted_time // 100)
        minutes = int(predicted_time % 100)
        time_formatted = f"{hours:02d}:{minutes:02d}"

        time_of_day = 'Día' if 600 <= predicted_time < 1800 else 'Noche'

        pdf_path = os.path.join(BASE_DIR, 'prediction_report.pdf')
        generate_pdf(time_formatted, time_of_day, pdf_path)

        return send_file(
                pdf_path,
                as_attachment=True,
                mimetype='application/pdf'
            )


    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
