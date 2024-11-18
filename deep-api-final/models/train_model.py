import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Rutas de los archivos
data_path = 'deep-api-final/data/processed/crime_data_clean.csv'  # Dataset limpio
model_path = 'deep-api-final/models/modelo_time_occ.pkl'  # Modelo entrenado
encoder_area_path = 'deep-api-final/app/enconders/encoder_area.pkl'  # Codificador para AREA NAME
encoder_crm_path = 'deep-api-final/app/enconders/encoder_crm.pkl'  # Codificador para Crm Cd Desc
encoder_premis_path = 'deep-api-final/app/enconders/encoder_premis.pkl'  # Codificador para Premis Desc
encoder_day_night_path = 'deep-api-final/app/enconders/encoder_day_night.pkl'  # Codificador para Day/Night

# 1. Cargar el dataset limpio
data = pd.read_csv(data_path)

# 2. Variables independientes (X) y dependiente (y)
X = data[['AREA NAME', 'Crm Cd Desc', 'Premis Desc', 'day_night']]  # Variables categóricas
y = data['TIME OCC']  # Variable dependiente (hora del crimen)

# 3. Codificar variables categóricas
encoder_area = LabelEncoder()
encoder_crm = LabelEncoder()
encoder_premis = LabelEncoder()
encoder_day_night = LabelEncoder()

# Entrenar los codificadores con todos los valores únicos
encoder_area.fit(data['AREA NAME'].unique())
encoder_crm.fit(data['Crm Cd Desc'].unique())
encoder_premis.fit(data['Premis Desc'].unique())
encoder_day_night.fit(data['day_night'].unique())

# Transformar las columnas categóricas
X['AREA NAME'] = encoder_area.transform(X['AREA NAME'])
X['Crm Cd Desc'] = encoder_crm.transform(X['Crm Cd Desc'])
X['Premis Desc'] = encoder_premis.transform(X['Premis Desc'])
X['day_night'] = encoder_day_night.transform(X['day_night'])

# 4. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar el modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Error Absoluto Medio (MAE): {mae:.2f}")

# 7. Guardar el modelo y los codificadores
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    print(f"Modelo guardado en: {model_path}")

with open(encoder_area_path, 'wb') as f:
    pickle.dump(encoder_area, f)
    print(f"Codificador AREA NAME guardado en: {encoder_area_path}")

with open(encoder_crm_path, 'wb') as f:
    pickle.dump(encoder_crm, f)
    print(f"Codificador Crm Cd Desc guardado en: {encoder_crm_path}")

with open(encoder_premis_path, 'wb') as f:
    pickle.dump(encoder_premis, f)
    print(f"Codificador Premis Desc guardado en: {encoder_premis_path}")

with open(encoder_day_night_path, 'wb') as f:
    pickle.dump(encoder_day_night, f)
    print(f"Codificador Day/Night guardado en: {encoder_day_night_path}")
