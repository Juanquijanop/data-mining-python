import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Paso 1: Cargar los datos
data = pd.read_csv('data.csv', low_memory=False)

# Asegurar que no haya espacios en los nombres de columnas
data.columns = data.columns.str.strip()

# Paso 2: Preprocesamiento de datos
# Eliminar columnas innecesarias o que no sean útiles para la predicción
data = data.drop(columns=['CASE#', 'FBI CD', 'X COORDINATE', 'Y COORDINATE', 'LATITUDE',
                          'LONGITUDE', 'LOCATION'])

# Separar las etiquetas antes de aplicar get_dummies
y = data['PRIMARY DESCRIPTION']
X = data.drop(columns=['PRIMARY DESCRIPTION'])

# Codificar las variables categóricas en X utilizando get_dummies
X = pd.get_dummies(X, drop_first=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# Paso 4: Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Paso 5: Guardar el modelo entrenado
joblib.dump(model, 'crime_prediction_model.pkl')
