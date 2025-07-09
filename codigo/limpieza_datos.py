import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ClienteTelco:
    """
    Clase que representa un cliente de telecomunicaciones con información relevante 
    para análisis de churn (abandono).

    Atributos:
        cliente_id (int): Identificador único del cliente.
        tenure (int): Meses de permanencia del cliente.
        monthly_charge (float): Cargo mensual en dólares.
        churn (int): Indicador de cancelación (1 si se va, 0 si se queda).
    """
    
    def __init__(self, cliente_id, tenure, monthly_charge, churn):
        """
        Inicializa un objeto ClienteTelco con sus atributos.

        Parametros:
        ----------------
            cliente_id (int): ID único del cliente.
            tenure (int): Tiempo de permanencia en meses.
            monthly_charge (float): Cargo mensual en dólares.
            churn (int): 1 si el cliente se va, 0 si se queda.
        """
        self._cliente_id = cliente_id
        self._tenure = tenure
        self._monthly_charge = monthly_charge
        self._churn = churn

    def __str__(self):
        """
        Retorna una representación en texto legible del objeto.

        Returns:
        -------------
            str: Descripción del cliente con ID, estado, tenure y cargo mensual.
        """
        estado = "Se va" if self._churn == 1 else "Se queda"
        return f"Cliente {self._cliente_id} - {estado} - Tenure: {self._tenure} meses - Cargo mensual: ${self._monthly_charge:.2f}"

    # Get y set para cada atributo
    @property
    def cliente_id(self):
        """
        Obtiene el ID del cliente.

        Returns:
        -----------------
            int: ID del cliente.
        """
        return self._cliente_id

    @cliente_id.setter
    def cliente_id(self, valor):
         """
        Establece un nuevo ID para el cliente.

        Parametros:
        -------------
            valor (int): Nuevo ID del cliente.
        """
        self._cliente_id = valor

    @property
    def tenure(self):
        """
        Obtiene el tiempo de permanencia del cliente.

        Returns:
        ----------------
            int: Meses de permanencia.
        """
        return self._tenure

    @tenure.setter
    def tenure(self, valor):
        """
        Establece un nuevo tiempo de permanencia.

        Parametros:
        ------------------
            valor (int): Nuevos meses de permanencia.
        """
        self._tenure = valor

    @property
    def monthly_charge(self):
        """
        Obtiene el cargo mensual del cliente.

        Returns:
        ----------------
            float: Cargo mensual en dólares.
        """
        return self._monthly_charge

    @monthly_charge.setter
    def monthly_charge(self, valor):
        """
        Establece un nuevo cargo mensual.

        Parametros:
        --------------
            valor (float): Nuevo cargo mensual en dólares.
        """
        self._monthly_charge = valor

    @property
    def churn(self):
        """
        Obtiene el estado de cancelación.

        Returns:
        ---------------
            int: 1 si el cliente se va, 0 si se queda.
        """
        return self._churn

    @churn.setter
    def churn(self, valor):
        """
        Establece un nuevo estado de cancelación.

        Parametros:
        ------------
            valor (int): 1 si se va, 0 si se queda.
        """
        self._churn = valor



# Cargar el dataset
df = pd.read_csv("../datos/Telco-Customer-Churn.csv")

# Eliminar columnas no útiles para el modelo
# 'customerID' es un identificador que no aporta información predictiva
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convertir 'TotalCharges' a numérico; si hay errores (como espacios vacíos), se convierten a NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Eliminar filas con datos faltantes (NaN), por ejemplo, donde TotalCharges no se pudo convertir
df = df.dropna()

# Convertir la variable objetivo 'Churn' a formato binario: 0 (No) y 1 (Yes)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Codificación de todas las variables categóricas
# Excluimos 'Churn' porque ya está en formato binario y no necesita codificarse
df_encoded = pd.get_dummies(df.drop('Churn', axis=1))

# Agregar de nuevo la variable objetivo al dataset codificado
df_encoded['Churn'] = df['Churn']


# Separar variables predictoras (X) y variable objetivo (y)
x = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Dividir el dataset en entrenamiento (70%) y prueba (30%)
# random_state permite que los resultados sean reproducibles
# stratify mantiene la proporción de clases igual en ambos conjuntos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=22, stratify=y
)
