
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv("C:/Users/Lenovo/Downloads/Telco-Customer-Churn4.csv")




X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]


class XGBModel:
    """
    Clasificador XGBoost encapsulado.

    Atributos
    ---------
    params        : dict   hiperparámetros
    test_size     : float  % validación
    random_state  : int    semilla
    val_score     : float  exactitud en validación
    """
    def __init__(self, params=None, test_size=0.3, random_state=42):
        base = {"n_estimators": 200, "learning_rate": 0.1,
                "max_depth": 5, "subsample": 0.8,
                "colsample_bytree": 0.8}
        self._params = base if params is None else params
        self._test_size = test_size
        self._random_state = random_state
        self.val_score = None
        self._model = XGBClassifier(**self._params,
                                    use_label_encoder=False,
                                    eval_metric="logloss")

    
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, nuevos):
        self._params = nuevos
        self._model = XGBClassifier(**nuevos,
                                    use_label_encoder=False,
                                    eval_metric="logloss")

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, v):
        self._test_size = v

  
    def __str__(self):
        acc = f"{self.val_score:.3f}" if self.val_score else "NA"
        return f"XGBModel(val_acc={acc})"

    
    def fit(self, X, y):
        '''Entrena el modelo con los datos proporcionados y calcula la exactitud en el conjunto de validación.

        Parámetros
        --------------------------
        X : pandas.DataFrame
            Características de los datos.
        y : pandas.Series
            Etiquetas de los datos.

        Retorna
        -------------------------
        self : XGBModel
            El objeto XGBModel entrenado.
        '''
        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=self._test_size,
            random_state=self._random_state, stratify=y)
        self._model.fit(Xtr, ytr)
        self.val_score = accuracy_score(yv, self._model.predict(Xv))
        return self

    def predict(self, X):
        '''Realiza predicciones sobre los datos dados usando el modelo entrenado.

        Parámetros
        --------------------------
        X : pandas.DataFrame
            Características de los datos para predecir.

        Retorna
        -------------------------
        preds : numpy.ndarray
            Predicciones del modelo.
        '''
        return self._model.predict(X)

    def evaluate(self, X, y):
        '''Evalúa el modelo en los datos proporcionados y muestra un informe de clasificación.

        Parámetros
        --------------------------
        X : pandas.DataFrame
            Características de los datos.
        y : pandas.Series
            Etiquetas de los datos.

        Retorna
        -------------------------
        acc : float
            Exactitud del modelo en los datos proporcionados.
        '''
        preds = self.predict(X)
        print(classification_report(y, preds))
        return accuracy_score(y, preds)

    def save(self, ruta="xgb.pkl"):
        '''Guarda el modelo entrenado en un archivo.

        Parámetros
        --------------------------
        ruta : str, opcional
            Ruta del archivo donde guardar el modelo.

        Retorna
        -------------------------
        None
        '''
        joblib.dump(self._model, ruta)

    def load(self, ruta="xgb.pkl"):
        '''Carga un modelo previamente guardado desde un archivo.

        Parámetros
        --------------------------
        ruta : str, opcional
            Ruta del archivo desde donde cargar el modelo.

        Retorna
        -------------------------
        self : XGBModel
            El objeto XGBModel con el modelo cargado.
        '''
        self._model = joblib.load(ruta)
        return self



modelo = XGBModel().fit(X, y)
print(modelo)                         
test_acc = modelo.evaluate(X, y)      
print("Accuracy :", round(modelo.val_score, 3))


