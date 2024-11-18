import pandas as pd
import numpy as np

# Cargar el dataset
file_path = 'data/raw/crime_data.csv'  # Ruta del archivo original
output_path = 'data/processed/crime_data_clean.csv'  # Ruta del archivo limpio
data = pd.read_csv(file_path)

# 1. Seleccionar columnas relevantes
columns_to_keep = [
    'AREA NAME',      # Nombre del área
    'Crm Cd Desc',    # Tipo de crimen
    'Premis Desc',    # Descripción del lugar
    'TIME OCC'        # Hora del crimen
]
data = data[columns_to_keep]

# 2. Eliminar filas con valores nulos
data = data.dropna()

# 3. Estandarizar columnas
# Convertir nombres a mayúsculas (opcional, según formato del dataset)
data['AREA NAME'] = data['AREA NAME'].str.upper()
data['Crm Cd Desc'] = data['Crm Cd Desc'].str.upper()
data['Premis Desc'] = data['Premis Desc'].str.upper()

# 4. Limitar rango de horas (opcional, por limpieza)
data = data[(data['TIME OCC'] >= 0) & (data['TIME OCC'] <= 2400)]

# 5. Eliminar duplicados
data = data.drop_duplicates()

# 6. Guardar dataset limpio
data.to_csv(output_path, index=False)

print(f"Archivo limpio guardado en: {output_path}")
