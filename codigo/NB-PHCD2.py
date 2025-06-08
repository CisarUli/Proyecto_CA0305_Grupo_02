import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ClienteTelco:
    def __init__(self, cliente_id, tenure, monthly_charge, churn):
        self._cliente_id = cliente_id
        self._tenure = tenure
        self._monthly_charge = monthly_charge
        self._churn = churn

    def __str__(self):
        estado = "Se va" if self._churn == 1 else "Se queda"
        return f"Cliente {self._cliente_id} - {estado} - Tenure: {self._tenure} meses - Cargo mensual: ${self._monthly_charge:.2f}"

    # Get y set para cada atributo
    @property
    def cliente_id(self):
        return self._cliente_id

    @cliente_id.setter
    def cliente_id(self, valor):
        self._cliente_id = valor

    @property
    def tenure(self):
        return self._tenure

    @tenure.setter
    def tenure(self, valor):
        self._tenure = valor

    @property
    def monthly_charge(self):
        return self._monthly_charge

    @monthly_charge.setter
    def monthly_charge(self, valor):
        self._monthly_charge = valor

    @property
    def churn(self):
        return self._churn

    @churn.setter
    def churn(self, valor):
        self._churn = valor



# Cargar el dataset
df = pd.read_csv("../datos/Telco-Customer-Churn4.csv")

# Eliminar columnas no útiles para el modelo
# 'customerID' es un identificador que no aporta información predictiva
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convertir 'TotalCharges' a numérico; si hay errores (como espacios vacíos), se convierten a NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eliminar filas con datos faltantes (NaN), por ejemplo, donde TotalCharges no se pudo convertir
df = df.dropna()

# Convertir la variable objetivo 'Churn' a formato binario: 0 (No) y 1 (Yes)
#Esta linea se comenta debido a un cambio en la base de datos, por lo que ya no es necesaria
# df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}) 

# Codificación de todas las variables categóricas
# Excluimos 'Churn' porque ya está en formato binario y no necesita codificarse
df_encoded = pd.get_dummies(df.drop('Churn', axis=1))

# Agregar de nuevo la variable objetivo al dataset codificado
df_encoded['Churn'] = df['Churn']


# Separar variables predictoras (X) y variable objetivo (y)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Dividir el dataset en entrenamiento (70%) y prueba (30%)
# random_state permite que los resultados sean reproducibles
# stratify mantiene la proporción de clases igual en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=22, stratify=y
)

# Crear el modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los valores con el conjunto de prueba
y_pred = modelo.predict(X_test)


# Mostrar la exactitud del modelo (accuracy): porcentaje de aciertos
print("Accuracy:", accuracy_score(y_test, y_pred))

# Mostrar la matriz de confusión: permite analizar errores y aciertos por clase
print("\nMatriz de confusión:")

#Ayuda a ver si el modelo se equivoca más al predecir que alguien no se va o que sí se va.
#(True Positives): 477 — predijo churn correctamente.

#(False Positives): 539 — dijo que se iba, pero no era cierto.

#(False Negatives): 84 — se fue, pero no lo detectó.

#(True Negatives): 1010 — predijo correctamente que no se iba.

print(confusion_matrix(y_test, y_pred))

# Mostrar métricas de evaluación por clase: precisión, recall y F1-score
print("\nReporte de clasificación:")

#Clase 0 (No churn):
#Precisión: 0.92  El modelo casi nunca se equivoca cuando predice que el cliente no se va.

#Recall: 0.65  Detecta el 65% de los que realmente no se van.

#F1-score: 0.76  Equilibrio entre precisión y recall.

#Clase 1 (Churn):
#Precisión: 0.47  Cuando predice que se va, solo el 47% es cierto.

#Recall: 0.85  Captura el 85% de los clientes que efectivamente se van.

#F1-score: 0.60  Moderadamente útil para detectar churn.

print(classification_report(y_test, y_pred))


#informacion de como hacerlo de:
#https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#https://scikit-learn.org/stable/modules/naive_bayes.html
