
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------------------------------------------------
# 1. Cargar y preparar los datos Telco Churn
# ------------------------------------------------------------------
df = pd.read_csv("C:/Users/Lenovo/Downloads/Telco-Customer-Churn.csv")

df = df.drop(columns="customerID")                     # ID no predictivo
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

# ------------------------------------------------------------------
# 2. Definición del modelo OO con getters, setters y __str__
# ------------------------------------------------------------------
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
    def __init__(self, params=None, test_size=0.2, random_state=42):
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

    # ---------- getters / setters ----------
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

    # ---------- utilidad ----------
    def __str__(self):
        acc = f"{self.val_score:.3f}" if self.val_score else "NA"
        return f"XGBModel(val_acc={acc})"

    # ---------- API ----------
    def fit(self, X, y):
        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=self._test_size,
            random_state=self._random_state, stratify=y)
        self._model.fit(Xtr, ytr)
        self.val_score = accuracy_score(yv, self._model.predict(Xv))
        return self

    def predict(self, X):
        return self._model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        print(classification_report(y, preds))
        return accuracy_score(y, preds)

    def save(self, ruta="xgb.pkl"):
        joblib.dump(self._model, ruta)

    def load(self, ruta="xgb.pkl"):
        self._model = joblib.load(ruta)
        return self


# ------------------------------------------------------------------
# 3. Entrenamiento y evaluación
# ------------------------------------------------------------------
modelo = XGBModel().fit(X, y)
print(modelo)                         # muestra val_acc
test_acc = modelo.evaluate(X, y)      # métrica global (ejemplo rápido)
print("Accuracy total:", round(test_acc, 3))


