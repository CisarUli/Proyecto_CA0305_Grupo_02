import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class XGBModel:
    """
    Clase para encapsular un clasificador XGBoost para predicción de churn.

    Permite entrenar, evaluar, predecir, guardar y cargar un modelo XGBoost con hiperparámetros configurables.
    Utiliza validación mediante división de datos y calcula métricas de rendimiento como la exactitud.

    Argumentos:
        params (dict, opcional): Hiperparámetros para el modelo XGBoost. Por defecto, se usan valores base.
        test_size (float, opcional): Proporción de datos para el conjunto de validación (0 a 1). Por defecto, 0.3.
        random_state (int, opcional): Semilla para la reproducibilidad. Por defecto, 42.

    Atributos:
        params (dict): Hiperparámetros del modelo.
        test_size (float): Proporción de datos para validación.
        random_state (int): Semilla para reproducibilidad.
        val_score (float): Exactitud en el conjunto de validación.
        _model (XGBClassifier): Instancia del modelo XGBoost.

    Métodos:
        fit(X, y): Entrena el modelo con los datos proporcionados.
        predict(X): Realiza predicciones con el modelo entrenado.
        evaluate(X, y): Evalúa el modelo y genera un informe de clasificación.
        save(ruta): Guarda el modelo entrenado en un archivo.
        load(ruta): Carga un modelo desde un archivo.
    """

    def __init__(self, params=None, test_size=0.3, random_state=42):
        """
        Inicializa una instancia de la clase XGBModel con hiperparámetros y configuración.

        Argumentos:
            params (dict, opcional): Hiperparámetros para XGBoost. Si es None, se usan valores base:
                - n_estimators: 200
                - learning_rate: 0.1
                - max_depth: 5
                - subsample: 0.8
                - colsample_bytree: 0.8
            test_size (float, opcional): Proporción del conjunto de validación (0 a 1). Por defecto, 0.3.
            random_state (int, opcional): Semilla para reproducibilidad. Por defecto, 42.

        Excepciones:
            ValueError: Si test_size no está entre 0 y 1.
        """
        if not 0 <= test_size <= 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        base = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        self._params = base if params is None else params
        self._test_size = test_size
        self._random_state = random_state
        self.val_score = None
        self._model = XGBClassifier(
            **self._params,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    @property
    def params(self):
        """
        Obtiene los hiperparámetros del modelo.

        Retorna:
            dict: Hiperparámetros actuales del modelo.
        """
        return self._params

    @params.setter
    def params(self, nuevos):
        """
        Establece nuevos hiperparámetros para el modelo y actualiza la instancia de XGBoost.

        Argumentos:
            nuevos (dict): Nuevos hiperparámetros para el modelo.

        Excepciones:
            TypeError: Si nuevos no es un diccionario.
        """
        if not isinstance(nuevos, dict):
            raise TypeError("Los hiperparámetros deben ser un diccionario.")
        self._params = nuevos
        self._model = XGBClassifier(
            **nuevos,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    @property
    def test_size(self):
        """
        Obtiene la proporción del conjunto de validación.

        Retorna:
            float: Proporción de datos para validación.
        """
        return self._test_size

    @test_size.setter
    def test_size(self, v):
        """
        Establece la proporción del conjunto de validación con validación.

        Argumentos:
            v (float): Nueva proporción para el conjunto de validación.

        Excepciones:
            ValueError: Si v no está entre 0 y 1.
        """
        if not 0 <= v <= 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        self._test_size = v

    def __str__(self):
        """
        Devuelve una representación en cadena de la instancia.

        Retorna:
            str: Cadena con la exactitud en validación (o 'NA' si no está entrenado).
        """
        acc = f"{self.val_score:.3f}" if self.val_score else "NA"
        return f"XGBModel(val_acc={acc})"

    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados y calcula la exactitud en el conjunto de validación.

        Divide los datos en conjuntos de entrenamiento y validación, entrena el modelo y almacena
        la exactitud en el conjunto de validación.

        Argumentos:
            X (pandas.DataFrame): Características de los datos.
            y (pandas.Series): Etiquetas de los datos.

        Retorna:
            XGBModel: El objeto XGBModel entrenado.

        Excepciones:
            ValueError: Si X o y están vacíos o no son del tipo esperado.
        """
        if X.empty or y.empty:
            raise ValueError("Los datos X e y no pueden estar vacíos.")
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X debe ser un DataFrame y y una Series de pandas.")
        Xtr, Xv, ytr, yv = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y
        )
        self._model.fit(Xtr, ytr)
        self.val_score = accuracy_score(yv, self._model.predict(Xv))
        return self

    def predict(self, X):
        """
        Realiza predicciones sobre los datos dados usando el modelo entrenado.

        Argumentos:
            X (pandas.DataFrame): Características de los datos para predecir.

        Retorna:
            numpy.ndarray: Predicciones del modelo.

        Excepciones:
            ValueError: Si X está vacío o no es un DataFrame.
            AttributeError: Si el modelo no ha sido entrenado.
        """
        if X.empty:
            raise ValueError("Los datos X no pueden estar vacíos.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas.")
        if self._model is None:
            raise AttributeError("El modelo no ha sido entrenado.")
        return self._model.predict(X)

    def evaluate(self, X, y):
        """
        Evalúa el modelo en los datos proporcionados y muestra un informe de clasificación.

        Calcula predicciones, genera un informe de clasificación (precisión, recall, f1-score)
        y retorna la exactitud.

        Argumentos:
            X (pandas.DataFrame): Características de los datos.
            y (pandas.Series): Etiquetas de los datos.

        Retorna:
            float: Exactitud del modelo en los datos proporcionados.

        Excepciones:
            ValueError: Si X o y están vacíos o no son del tipo esperado.
        """
        if X.empty or y.empty:
            raise ValueError("Los datos X e y no pueden estar vacíos.")
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("X debe ser un DataFrame y y una Series de pandas.")
        preds = self.predict(X)
        print(classification_report(y, preds))
        return accuracy_score(y, preds)

    def save(self, ruta="xgb.pkl"):
        """
        Guarda el modelo entrenado en un archivo.

        Argumentos:
            ruta (str, opcional): Ruta del archivo donde guardar el modelo. Por defecto, 'xgb.pkl'.

        Retorna:
            None

        Excepciones:
            AttributeError: Si el modelo no ha sido entrenado.
            OSError: Si hay un error al guardar el archivo.
        """
        if self._model is None:
            raise AttributeError("El modelo no ha sido entrenado.")
        try:
            joblib.dump(self._model, ruta)
        except OSError as e:
            raise OSError(f"Error al guardar el modelo en {ruta}: {e}")

    def load(self, ruta="xgb.pkl"):
        """
        Carga un modelo previamente guardado desde un archivo.

        Argumentos:
            ruta (str, opcional): Ruta del archivo desde donde cargar el modelo. Por defecto, 'xgb.pkl'.

        Retorna:
            XGBModel: El objeto XGBModel con el modelo cargado.

        Excepciones:
            FileNotFoundError: Si el archivo no existe.
            OSError: Si hay un error al cargar el archivo.
        """
        try:
            self._model = joblib.load(ruta)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo {ruta}.")
        except OSError as e:
            raise OSError(f"Error al cargar el modelo desde {ruta}: {e}")
        return self


if __name__ == "__main__":
    
    df = pd.read_csv("Telco-Customer-Churn4.csv")
    X = pd.get_dummies(df.drop("Churn", axis=1))
    y = df["Churn"]

    
    modelo = XGBModel().fit(X, y)
    print(modelo)  
    test_acc = modelo.evaluate(X, y)  
    print("Accuracy:", round(modelo.val_score, 3))  


