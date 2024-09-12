import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta

##################POINT 1 UPLOAD DATA #######################

# Cargar el archivo CSV correctamente
df = pd.read_csv('LACrimesData.csv', low_memory=False)



##################POINT 2 FILTER BY COLUM ###################


# Filtrar los delitos más graves usando la columna 'Crm Cd 1'
codigos_graves = [110, 210, 310, 410, 510, 610]
delitos_graves = df[df['Crm Cd 1'].isin(codigos_graves)]

#print(delitos_graves.head())



##################POINT 3 GROUPS BY #########################

# Contar el número de incidentes por tipo de delito (Crm Cd)
incidentes_por_tipo_delito = df.groupby('Crm Cd').size()
#print(incidentes_por_tipo_delito)


# Contar el número de incidentes por estatus de investigación (Status)
incidentes_por_status = df.groupby('Status').size()
#print(incidentes_por_status)


# Contar el número de incidentes por género de la víctima
incidentes_por_genero = df.groupby('Vict Sex').size()
#print(incidentes_por_genero)


# Contar el número de incidentes por arma utilizada
incidentes_por_arma = df.groupby('Weapon Used Cd').size()
#print(incidentes_por_arma)


# Contar el número de incidentes por edad de la víctima
incidentes_por_edad = df.groupby('Vict Age').size()
#print(incidentes_por_edad)


##################POINT 4 COUNTS #########################


# Contar los incidentes según la edad de la víctima
delitos_por_edad = df['Vict Age'].value_counts()
#print(delitos_por_edad)

# Contar los incidentes según el área geográfica
delitos_por_area = df['AREA'].value_counts()
#print(delitos_por_area)


# Contar los incidentes según el estatus de investigación
delitos_por_status = df['Status'].value_counts()
#print(delitos_por_status)


# Contar los incidentes según el tipo de crimen
delitos_por_tipo = df['Crm Cd'].value_counts()
#print(delitos_por_tipo)


####################POINT 5 CHARTS#########################



# Agrupar por tipo de crimen y contar el número de incidentes
incidentes_por_crimen = df.groupby('Crm Cd').size()

# Crear el gráfico de barras
incidentes_por_crimen.plot(kind='bar', figsize=(10, 6))
plt.title('Número de incidentes por tipo de crimen')
plt.xlabel('Código del crimen')
plt.ylabel('Número de incidentes')

#plt.savefig('point-5-photos/crmcd.png')



# Agrupar por género de la víctima y contar el número de incidentes
incidentes_por_genero = df.groupby('Vict Sex').size()

# Crear el gráfico de barras
incidentes_por_genero.plot(kind='bar', figsize=(10, 6))
plt.title('Número de incidentes por género de la víctima')
plt.xlabel('Género')
plt.ylabel('Número de incidentes')

# Guardar el gráfico como imagen PNG
#plt.savefig('point-5-photos/vict_sex.png')

# Agrupar por estatus de investigación y contar el número de incidentes
incidentes_por_status = df.groupby('Status').size()

# Crear el gráfico de barras
incidentes_por_status.plot(kind='bar', figsize=(10, 6))
plt.title('Número de incidentes por estatus de investigación')
plt.xlabel('Estatus de investigación')
plt.ylabel('Número de incidentes')

# Guardar el gráfico como imagen PNG
#plt.savefig('point-5-photos/status.png')


# Agrupar por edad de la víctima y contar el número de incidentes
incidentes_por_edad = df.groupby('Vict Age').size()

# Crear el gráfico de barras
incidentes_por_edad.plot(kind='bar', figsize=(10, 6))
plt.title('Número de incidentes por edad de la víctima')
plt.xlabel('Edad')
plt.ylabel('Número de incidentes')

# Guardar el gráfico como imagen PNG
#plt.savefig('point-5-photos/vict_age.png')


# Agrupar por tipo de arma utilizada y contar el número de incidentes
incidentes_por_arma = df.groupby('Weapon Used Cd').size()

# Crear el gráfico de barras
incidentes_por_arma.plot(kind='bar', figsize=(10, 6))
plt.title('Número de incidentes por tipo de arma utilizada')
plt.xlabel('Código de arma utilizada')
plt.ylabel('Número de incidentes')

# Guardar el gráfico como imagen PNG
#plt.savefig('point-5-photos/weapon_used.png')

########################POINT 6  GEO ####################################

# Asegúrate de que las columnas LAT y LON sean numéricas
df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
df['LON'] = pd.to_numeric(df['LON'], errors='coerce')

# Eliminar filas con valores NaN en LAT y LON
df = df.dropna(subset=['LAT', 'LON'])

# Crear geometría de puntos
df['geometry'] = df.apply(lambda row: Point(row['LON'], row['LAT']), axis=1)

# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Cargar el shapefile correctamente
world = gpd.read_file('ne_110m_admin_0_countries.shp')

# Filtrar y visualizar el mapa (ajusta el nombre de la columna según corresponda)
ax = world[world['NAME'] == 'United States of America'].plot(color='white', edgecolor='black')

# Ajustar el aspecto del gráfico
ax.set_aspect('auto')

# Ajustar límites para Los Ángeles
ax.set_xlim([-119, -117]) 
ax.set_ylim([33, 35]) 

# Plotear los incidentes
gdf.plot(ax=ax, marker='o', color='red', markersize=5)

# Guardar el mapa como imagen PNG
#plt.savefig('point-6-photos/mapa_incidentes.png')

###########################POINT 7 FILTER DATES################


#Modificacion a Columna DATE OCC para formatear la fecha
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

# Convertir la columna de fecha de ocurrencia a formato datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y')

# Filtrar incidentes ocurridos el 10 de mayo de 2023
incidentes_dia = df[df['DATE OCC'] == '2023-05-10']

# Mostrar los resultados
#print(incidentes_dia.head())


# Convertir la columna de fecha de ocurrencia a formato datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y')

# Filtrar incidentes ocurridos entre el 1 de enero y el 31 de diciembre de 2020
rango_fechas = (df['DATE OCC'] >= '2020-01-01') & (df['DATE OCC'] <= '2020-12-31')
incidentes_en_rango = df[rango_fechas]

# Mostrar los resultados
#print(incidentes_en_rango.head())




# Fecha actual
hoy = datetime.now()

# Filtrar incidentes ocurridos en los últimos 30 días
incidentes_ultimos_30_dias = df[df['DATE OCC'] >= (hoy - timedelta(days=30))]

# Mostrar los resultados
#print(incidentes_ultimos_30_dias.head())


#######################POINT 8 FRECUENCY################


# Contar el número de incidentes por tipo de crimen
delitos_por_tipo = df['Crm Cd Desc'].value_counts()

# Mostrar los resultados
#print(delitos_por_tipo)


# Contar el número de incidentes por cada área
delitos_por_area = df['AREA NAME'].value_counts()

# Mostrar los resultados
#print(delitos_por_area)


# Contar el número de incidentes por género de la víctima
delitos_por_genero = df['Vict Sex'].value_counts()

# Mostrar los resultados
#print(delitos_por_genero)


#######################POINT 9 COMMUN DATA################



# Filtrar los incidentes en el área de Hollywood
incidentes_hollywood = df[df['AREA NAME'] == 'Hollywood']

# Contar la frecuencia de cada tipo de delito en Hollywood
delitos_comunes_hollywood = incidentes_hollywood['Crm Cd Desc'].value_counts()

# Mostrar los 10 delitos más comunes en Hollywood
#print(delitos_comunes_hollywood.head(10))



# Filtrar los incidentes donde se utilizó un arma
incidentes_con_arma = df[df['Weapon Used Cd'].notna()]

# Contar la frecuencia de cada tipo de delito en incidentes con armas
delitos_con_arma = incidentes_con_arma['Crm Cd Desc'].value_counts()

# Mostrar los 10 delitos más comunes en incidentes con armas
#print(delitos_con_arma.head(10))



# Filtrar los incidentes donde la víctima es mujer (F)
incidentes_mujeres = df[df['Vict Sex'] == 'F']

# Contar la frecuencia de cada tipo de delito en incidentes donde la víctima es mujer
delitos_mujeres = incidentes_mujeres['Crm Cd Desc'].value_counts()

# Mostrar los 10 delitos más comunes en incidentes con víctimas mujeres
#print(delitos_mujeres.head(10))


#########################POINT 10 DISTRIBUTION DATA ##########################


# Convertir Vict Age a numérico, ignorando valores que no se pueden convertir
df['Vict Age'] = pd.to_numeric(df['Vict Age'], errors='coerce')

# Visualizar la distribución de edades por género con un histograma
df[df['Vict Sex'] == 'M']['Vict Age'].plot(kind='hist', bins=20, alpha=0.5, label='Masculino', figsize=(10, 6))
df[df['Vict Sex'] == 'F']['Vict Age'].plot(kind='hist', bins=20, alpha=0.5, label='Femenino')

plt.title('Distribución de edades de las víctimas por género')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.legend()

# Guardar la imagen
plt.savefig('point-10-photos/distribucion_edades_genero.png')


# Agrupar por tipo de crimen y calcular la edad promedio de las víctimas
edad_promedio_por_crimen = df.groupby('Crm Cd Desc')['Vict Age'].mean().nlargest(10)

# Crear el gráfico de barras
edad_promedio_por_crimen.plot(kind='bar', figsize=(10, 6))
plt.title('Edad promedio de las víctimas por tipo de crimen')
plt.xlabel('Tipo de crimen')
plt.ylabel('Edad promedio')

# Guardar la imagen
plt.savefig('point-10-photos/edad_promedio_por_crimen.png')

# Filtrar los incidentes con víctimas menores de edad
victimas_menores = df[df['Vict Age'] < 18]

# Contar el número de incidentes por tipo de crimen para víctimas menores de edad
incidentes_menores_por_crimen = victimas_menores['Crm Cd Desc'].value_counts().nlargest(10)

# Crear el gráfico de barras
incidentes_menores_por_crimen.plot(kind='bar', figsize=(10, 6))
plt.title('Incidentes con víctimas menores de edad por tipo de crimen')
plt.xlabel('Tipo de crimen')
plt.ylabel('Número de incidentes')

# Guardar la imagen
plt.savefig('point-10-photos/incidentes_menores_por_crimen.png')
