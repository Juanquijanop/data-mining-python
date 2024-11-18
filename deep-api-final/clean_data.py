import pandas as pd
import numpy as np

file_path = 'deep-api-final/data/raw/LACrimesData.csv'
output_path = 'deep-api-final/data/processed/crime_data_clean.csv'

# 1. Cargar el dataset
data = pd.read_csv(file_path)

# 2. Seleccionar columnas relevantes
columns_to_keep = [
    'AREA NAME',      # Nombre del área
    'Crm Cd Desc',    # Tipo de crimen
    'Premis Desc',    # Descripción del lugar
    'TIME OCC'        # Hora del crimen
]
data = data[columns_to_keep]

# 3. Eliminar filas con valores nulos
data = data.dropna()

# 4. Estandarizar columnas
# Convertir nombres a mayúsculas
data['AREA NAME'] = data['AREA NAME'].str.upper()
data['Crm Cd Desc'] = data['Crm Cd Desc'].str.upper()
data['Premis Desc'] = data['Premis Desc'].str.upper()

# 5. Limitar rango de horas (opcional, por limpieza)
data = data[(data['TIME OCC'] >= 0) & (data['TIME OCC'] <= 2400)]

# 6. Eliminar duplicados
data = data.drop_duplicates()

# 7. Crear la variable day_night
def assign_day_night(hour):
    """
    Clasifica las horas como Día o Noche.
    """
    if 600 <= hour < 1800:
        return 'Día'
    else:
        return 'Noche'

data['day_night'] = data['TIME OCC'].apply(assign_day_night)

# 8. Guardar el dataset limpio
data.to_csv(output_path, index=False)

print(f"{output_path}")
