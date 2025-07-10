import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class NB:
    """
    Clase para entrenar y evaluar un modelo Naive Bayes Gaussiano sobre un dataset.

    Parametros
    -----------
        X (DataFrame): Variables predictoras.
        y (Series): Variable objetivo binaria.
        test_size (float): Proporción de datos para prueba (default: 0.3).
        random_state (int): Semilla para reproducibilidad (default: 22).
    """

    def __init__(self, X, y, test_size=0.1, random_state=42):
        """
       Inicializa la clase NB y divide el dataset.
        Parametros:
        -----------
        X (DataFrame): Variables predictoras.
        y (Series): Variable objetivo.
        test_size (float): Proporción de prueba.
        random_state (int): Semilla.
        """
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self._modelo = GaussianNB()
        self._y_pred = None
        self._accuracy = None

    def entrenar(self):
        """
        Entrena el modelo con los datos de entrenamiento.
        """
        self.modelo.fit(self.X_train, self.y_train)

    def predecir(self):
        """
        Realiza predicciones sobre el conjunto de prueba.

        Returns:
        ----------
            ndarray: Predicciones.
        """
        self.y_pred = self.modelo.predict(self.X_test)
        return self.y_pred

    def probar(self):
        """
        Entrena el modelo, predice y muestra métricas de evaluación.
        """
        
        self.entrenar()
        self.predecir()

        acc = accuracy_score(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)

        print(f"Accuracy: {acc:.4f}\n")
        print("Matriz de confusión:")
        print(cm)
        print("\nReporte de clasificación:")
        print(report)


    def __str__(self):
        """
        Retorna un resumen legible del modelo y resultados principales.

        Returns:
        ----------
            str: Resumen con accuracy y tamaños de conjuntos.
        """
        resumen = (
            f"Modelo Naive Bayes Gaussiano\n"
            f" - Tamaño entrenamiento: {self.X_train.shape[0]} muestras\n"
            f" - Tamaño prueba: {self.X_test.shape[0]} muestras\n"
        )
        if self.accuracy is not None:
            resumen += f" - Accuracy actual: {self.accuracy:.4f}\n"
        else:
            resumen += " - Modelo aún no ha sido probado.\n"
        return resumen

    #Getters y setters de la clase
    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, valor):
        self._X_train = valor

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, valor):
        self._X_test = valor

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, valor):
        self._y_train = valor

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, valor):
        self._y_test = valor

    @property
    def modelo(self):
        return self._modelo

    @modelo.setter
    def modelo(self, valor):
        self._modelo = valor

    @property
    def y_pred(self):
        return self._y_pred

    @y_pred.setter
    def y_pred(self, valor):
        self._y_pred = valor

    @property
    def accuracy(self):
        return self._accuracy

    @accuracy.setter
    def accuracy(self, valor):
        self._accuracy = valor


# Cargar el dataset
#df = pd.read_csv("../datos/Telco-Customer-Churn4.csv")

# Eliminar columnas no útiles para el modelo
# 'customerID' es un identificador que no aporta información predictiva
#if 'customerID' in df.columns:
  #  df = df.drop(columns=['customerID'])

# Convertir 'TotalCharges' a numérico; si hay errores (como espacios vacíos), se convierten a NaN
#df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eliminar filas con datos faltantes (NaN), por ejemplo, donde TotalCharges no se pudo convertir
#df = df.dropna()

# Convertir la variable objetivo 'Churn' a formato binario: 0 (No) y 1 (Yes)
#Esta linea se comenta debido a un cambio en la base de datos, por lo que ya no es necesaria
# df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1}) 

# Codificación de todas las variables categóricas
# Excluimos 'Churn' porque ya está en formato binario y no necesita codificarse
#df_encoded = pd.get_dummies(df.drop('Churn', axis=1))

# Agregar de nuevo la variable objetivo al dataset codificado
#df_encoded['Churn'] = df['Churn']


# Separar variables predictoras (X) y variable objetivo (y)
#X = df_encoded.drop("Churn", axis=1)
#y = df_encoded["Churn"]

# Dividir el dataset en entrenamiento (70%) y prueba (30%)
# random_state permite que los resultados sean reproducibles
# stratify mantiene la proporción de clases igual en ambos conjuntos
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=22, stratify=y
#)

# Crear el modelo Naive Bayes Gaussiano
#modelo = GaussianNB()

# Entrenar el modelo con los datos de entrenamiento
#modelo.fit(X_train, y_train)

# Predecir los valores con el conjunto de prueba
#y_pred = modelo.predict(X_test)


# Mostrar la exactitud del modelo (accuracy): porcentaje de aciertos
#print("Accuracy:", accuracy_score(y_test, y_pred))

# Mostrar la matriz de confusión: permite analizar errores y aciertos por clase
#print("\nMatriz de confusión:")

#Ayuda a ver si el modelo se equivoca más al predecir que alguien no se va o que sí se va.
#(True Positives): 477 — predijo churn correctamente.

#(False Positives): 539 — dijo que se iba, pero no era cierto.

#(False Negatives): 84 — se fue, pero no lo detectó.

#(True Negatives): 1010 — predijo correctamente que no se iba.

#print(confusion_matrix(y_test, y_pred))

# Mostrar métricas de evaluación por clase: precisión, recall y F1-score
#print("\nReporte de clasificación:")

#Clase 0 (No churn):
#Precisión: 0.92  El modelo casi nunca se equivoca cuando predice que el cliente no se va.

#Recall: 0.65  Detecta el 65% de los que realmente no se van.

#F1-score: 0.76  Equilibrio entre precisión y recall.

#Clase 1 (Churn):
#Precisión: 0.47  Cuando predice que se va, solo el 47% es cierto.

#Recall: 0.85  Captura el 85% de los clientes que efectivamente se van.

#F1-score: 0.60  Moderadamente útil para detectar churn.

#print(classification_report(y_test, y_pred))


#informacion de como hacerlo de:
#https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#https://scikit-learn.org/stable/modules/naive_bayes.html
