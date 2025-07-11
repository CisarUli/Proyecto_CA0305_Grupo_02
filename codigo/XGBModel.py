import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

class XGBModel:

    def __init__(self,  X, y, params=None, test_size=0.1, random_state=42):
        """
        Constructor por defecto de la clase.

        Parámetros
        ----------
        params : dict, opcional
            Hiperparámetros para XGBoost. Si es None, se usan valores base:
            - n_estimators: 200
            - learning_rate: 0.1
            - max_depth: 5
            - subsample: 0.8
            - colsample_bytree: 0.8
        test_size : float, opcional
            Proporción del conjunto de validación (0 a 1). Por defecto, 0.1.
        random_state : int, opcional
            Semilla para reproducibilidad. Por defecto, 42.

        Retorna
        -------
        None
        """
        self._X_train, self._X_test, self._var_pred_train, self._var_pred_test = train_test_split(
            X, y, test_size = test_size, stratify = y, random_state = random_state
        )
        
        if not 0 <= test_size <= 1:
            raise ValueError("test_size debe estar entre 0 y 1.")
        base = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        self._parametros = base if params is None else params
        self._tamano_prueba = test_size
        self._random_state = random_state
        self.puntuacion_val = None
        self._modelo = XGBClassifier(
            **self._parametros,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    @property
    def parametros(self):
        """
        Obtiene los hiperparámetros del modelo.

        Retorna:
            dict: Hiperparámetros actuales del modelo.
        """
        return self._parametros

    @parametros.setter
    def parametros(self, nuevos):
        """
        Establece nuevos hiperparámetros para el modelo y actualiza la instancia de XGBoost.

        Argumentos:
            nuevos (dict): Nuevos hiperparámetros para el modelo.

        Excepciones:
            TypeError: Si nuevos no es un diccionario.
        """
        if not isinstance(nuevos, dict):
            raise TypeError("Los hiperparámetros deben ser un diccionario.")
        self._parametros = nuevos
        self._modelo = XGBClassifier(
            **nuevos,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    @property
    def tamano_prueba(self):
        """
        Obtiene la proporción del conjunto de validación.

        Retorna:
            float: Proporción de datos para validación.
        """
        return self._tamano_prueba

    @tamano_prueba.setter
    def tamano_prueba(self, v):
        """
        Establece la proporción del conjunto de validación con validación.

        Argumentos:
            v (float): Nueva proporción para el conjunto de validación.

        Excepciones:
            ValueError: Si v no está entre 0 y 1.
        """
        if not 0 <= v <= 1:
            raise ValueError("tamano_prueba debe estar entre 0 y 1.")
        self._tamano_prueba = v
        
    @property
    def X_train(self):
        """
        Obtiene el conjunto de entrenamiento de variables predictorias.

        Retorna:
            pandas.DataFrame: El conjunto de entrenamiento de variables predictorias.
        """
        return self._X_train

    @X_train.setter
    def X_train(self, valor):
        """
        Establece el conjunto de entrenamiento de variables predictorias.

        Argumentos:
            valor (pandas.DataFrame): Nuevo conjunto de entrenamiento de variables predictorias.
        """
        self._X_train = valor

    @property
    def X_test(self):
        """
        Obtiene el conjunto de prueba de variables predictorias.

        Retorna:
            pandas.DataFrame: El conjunto de prueba de variables predictorias.
        """
        return self._X_test

    @X_test.setter
    def X_test(self, valor):
        """
        Establece el conjunto de prueba de variables predictorias.

        Argumentos:
            valor (pandas.DataFrame): Nuevo conjunto de prueba de variables predictorias.
        """
        self._X_test = valor

    @property
    def var_pred_train(self):
        """
        Obtiene el conjunto de entrenamiento de variable a predecir.

        Retorna:
            pandas.DataFrame: El conjunto de entrenamiento de variable a predecir.
        """
        return self._var_pred_train

    @var_pred_train.setter
    def var_pred_train(self, valor):
        """
        Establece el conjunto de entrenamiento de variable a predecir.

        Argumentos:
            valor (pandas.DataFrame): Nuevo conjunto de entrenamiento de variable a predecir.
        """
        self._var_pred_train = valor

    @property
    def var_pred_test(self):
        """
        Obtiene el conjunto de prueba de variable a predecir.

        Retorna:
            pandas.DataFrame: El conjunto de prueba de variable a predecir.
        """
        return self._var_pred_test

    @var_pred_test.setter
    def var_pred_test(self, valor):
        """
        Establece el conjunto de prueba de variable a predecir.

        Argumentos:
            valor (pandas.DataFrame): Nuevo conjunto de prueba de variable a predecir.
        """
        self._var_pred_test = valor

    def __str__(self):
        """
        Devuelve una representación en cadena de la instancia.

        Retorna:
            str: Cadena con la exactitud en validación (o 'NA' si no está entrenado).
        """
        acc = f"{self.puntuacion_val:.3f}" if self.puntuacion_val else "NA"
        return f"XGBModel(val_acc={acc})"

    def entrenar(self):
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
        if self._X_train.empty or self._var_pred_train.empty:
            raise ValueError("Los datos X e y no pueden estar vacíos.")
        if not isinstance(self._X_train, pd.DataFrame) or not isinstance(self._var_pred_train, pd.Series):
            raise ValueError("X debe ser un DataFrame y y una Series de pandas.")
        self._modelo.fit(self._X_train, self._var_pred_train)
        self.puntuacion_val = accuracy_score(self._var_pred_test, self._modelo.predict(self._X_test))
        return self

    def predecir(self):
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
        if self._X_test.empty:
            raise ValueError("Los datos X no pueden estar vacíos.")
        if not isinstance(self._X_test, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas.")
        if self._modelo is None:
            raise AttributeError("El modelo no ha sido entrenado.")
        return self._modelo.predict(self._X_test)

    def evaluar(self):
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
        if self._X_test.empty or self._var_pred_test.empty:
            raise ValueError("Los datos X e y no pueden estar vacíos.")
        if not isinstance(self._X_test, pd.DataFrame) or not isinstance(self._var_pred_test, pd.Series):
            raise ValueError("X debe ser un DataFrame y y una Series de pandas.")
        self.entrenar()
        preds = self.predecir()
        print(classification_report(self._var_pred_test, preds))
        return accuracy_score(self._var_pred_test, preds)

    def guardar(self, ruta="xgb.pkl"):
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
        if self._modelo is None:
            raise AttributeError("El modelo no ha sido entrenado.")
        try:
            joblib.dump(self._modelo, ruta)
        except OSError as e:
            raise OSError(f"Error al guardar el modelo en {ruta}: {e}")

    def cargar(self, ruta="xgb.pkl"):
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
            self._modelo = joblib.load(ruta)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo {ruta}.")
        except OSError as e:
            raise OSError(f"Error al cargar el modelo desde {ruta}: {e}")
        return self

