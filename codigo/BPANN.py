import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ActivityRegularization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class BPANN:
    def __init__(self, ruta_archivo):
        """Constructor por defecto de la clase.

        Parámetros
        ----------
        ruta_archivo:str
            La ruta al archivo con los datos.

        Retorna
        -------
        
        """
        self.__ruta_archivo = ruta_archivo
        self.__datos = None
        self.__datos_input = None
        self.__var_pred = None
        self.__modelo = None
        self.__X_train = None
        self.__X_test = None
        self.__var_pred_train = None
        self.__var_pred_test = None
        
    def __str__(self):
        """Descripción del objeto.
        
        Parámetros
        ----------

        Retorna
        -------
            Una descripción del objeto.
        """
        
        return f"Archivo con los datos: {self.__ruta_archivo}\nDatos\n: {self.__datos}\n \
            Datos limpios para la capa input:\n{self.__datos_input}\nVariable a predecir: Churn\
                \nDatos usados para entrenamiento y prueba:\nEntrenamiento\n{self.__X_train}\n{self.__var_pred_train}\n\
                    Prueba:\n{self.__X_test}\n{self.__var_pred_test}"

    def cargar_datos(self):
        """Carga, limpia y ajusta los datos para el modelo.

        Parámetros
        ----------

        Retorna
        -------
        
        """
        self.__datos = pd.read_csv(self.__ruta_archivo)
        self.__datos = pd.get_dummies(self.__datos, columns = ['Contract', 'PaymentMethod'], drop_first = True)
        self.__var_pred = self.__datos['Churn'].astype(int)
        datos_input = self.__datos.drop(columns = 'Churn')
        
        imputer = SimpleImputer(strategy = 'mean')
        datos_imputados = pd.DataFrame(imputer.fit_transform(datos_input), columns = datos_input.columns)
        numeric_cols = datos_imputados.select_dtypes(include = ['float64', 'int64']).columns
        scaler = MinMaxScaler()
        datos_imputados[numeric_cols] = scaler.fit_transform(datos_imputados[numeric_cols])        
        lasso = LassoCV(cv = 5).fit(datos_imputados, self.__var_pred)
        selected_features = datos_imputados.columns[lasso.coef_ != 0]
        self.__datos_input = datos_imputados[selected_features]
        
        ros = RandomOverSampler(random_state = 42)
        inputs_balanceados, preds_balanceados = ros.fit_resample(self.__datos_input, self.__var_pred)
        
        self.__X_train, self.__X_test, self.__var_pred_train, self.__var_pred_test = train_test_split(
            inputs_balanceados, preds_balanceados, test_size = 0.1, stratify = preds_balanceados, random_state = 42
        )

    def definir_modelo(self):
        """Define el modelo que se va a utilizar, en este caso una red neuronal con retropropagación.

        Parámetros
        ----------

        Retorna
        -------
        
        """
        input_dim = self.__datos_input.shape[1]
        self.__modelo = Sequential([
            Dense(250, input_dim = input_dim, activation = 'relu'),
            ActivityRegularization(l1 = 0.01, l2 = 0.01),
            Dense(250, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])
        self.__modelo.compile(optimizer = Adam(), 
                              loss = 'binary_crossentropy', 
                              metrics = ['accuracy']
                              )

    def entrenar(self, epochs = 4000, batch_size = 32):
        """Ejecuta el modelo con una porcentaje de datos para ajustar pesos y sesgos, en este caso 90%, antes de predecir.

        Parámetros
        ----------
        epochs: int
            La cantidad de iteraciones de entrenamiento, el valor por defecto es el máximo, típicamente se detiene antes por el early stop.
        batch_size: int
            Cantidad de grupos a utilizar (separación de datos) para la optimización por epoch.

        Retorna
        -------
        
        """
        early_stop = EarlyStopping(monitor = 'val_loss', 
                                   patience = 100, 
                                   baseline = 0.36,
                                   restore_best_weights = True
                                   )
        checkpoint = ModelCheckpoint('best_model.h5', 
                                     monitor = 'val_loss', 
                                     save_best_only = True
                                     )

        self.__modelo.fit(
            self.__X_train, self.__var_pred_train,
            validation_split = 0.1,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [early_stop, checkpoint],
            verbose = 1
        )

    def evaluar(self):
        """Predice e imprime evaluaciones acerca de la predicción.

        Parámetros
        ----------

        Retorna
        -------
        
        """
        y_pred = (self.__modelo.predict(self.__X_test) > 0.5).astype(int)
        print("Matriz de confusión:")
        print(confusion_matrix(self.__var_pred_test, y_pred))
        print(classification_report(self.__var_pred_test, y_pred))
        print("AUC:", roc_auc_score(self.__var_pred_test, y_pred))

    def probar(self):
        """Ejecuta el modelo.

        Parámetros
        ----------

        Retorna
        -------
        
        """
        self.cargar_datos()
        self.definir_modelo()
        self.entrenar()
        self.evaluar()
    
    @property
    def ruta_archivo(self):
        """Valor del atributo ruta_archivo.

        Retorna
        -------
        El valor del atributo ruta_archivo.

        """
        return self.__ruta_archivo
    
    @ruta_archivo.setter
    def ruta_archivo(self, new_value):
        """Cambia el valor del atributo ruta_archivo.
        
        Parámetros
        ----------
            new_value: str
                El nuevo valor del atributo ruta_archivo.
        """
        self.__ruta_archivo = new_value
        
    @property
    def datos(self):
        """Valor del atributo datos.

        Retorna
        -------
        El valor del datos.

        """
        return self.__datos
    
    @datos.setter
    def datos(self, new_value):
        """Cambia el valor del atributo datos.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo datos.
        """
        self.__datos = new_value
        
    @property
    def datos_input(self):
        """Valor del atributo datos_input.

        Retorna
        -------
        El valor del atributo datos_input.

        """
        return self.__datos_input
    
    @datos_input.setter
    def datos_input(self, new_value):
        """Cambia el valor del atributo datos_input.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo datos_input.
        """
        self.__datos_input = new_value
        
    @property
    def var_pred(self):
        """Valor del atributo var_pred.

        Retorna
        -------
        El valor del atributo var_pred.

        """
        return self.__var_pred
    
    @var_pred.setter
    def var_pred(self, new_value):
        """Cambia el valor del atributo var_pred.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo var_pred.
        """
        self.__var_pred = new_value
        
    @property
    def modelo(self):
        """Valor del atributo modelo.

        Retorna
        -------
        El valor del atributo modelo.

        """
        return self.__modelo
    
    @modelo.setter
    def modelo(self, new_value):
        """Cambia el valor del atributo modelo.
        
        Parámetros
        ----------
            new_value: tensorflow.keras.models.Sequential
                El nuevo valor del atributo modelo.
        """
        self.__modelo = new_value
        
    @property
    def X_train(self):
        """Valor del atributo X_train.

        Retorna
        -------
        El valor del atributo X_train.

        """
        return self.__X_train
    
    @X_train.setter
    def X_train(self, new_value):
        """Cambia el valor del atributo X_train.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo X_train.
        """
        self.__X_train = new_value
    
    @property
    def X_test(self):
        """Valor del atributo X_test.

        Retorna
        -------
        El valor del atributo X_test.

        """
        return self.__X_test
    
    @X_test.setter
    def X_test(self, new_value):
        """Cambia el valor del atributo X_test.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo X_test.
        """
        self.__X_test = new_value
        
    @property
    def var_pred_train(self):
        """Valor del atributo var_pred_train.

        Retorna
        -------
        El valor del atributo var_pred_train.

        """
        return self.__var_pred_train
    
    @var_pred_train.setter
    def var_pred_train(self, new_value):
        """Cambia el valor del atributo var_pred_train.
        
        Parámetros
        ----------
            new_value: str
                El nuevo valor del atributo var_pred_train.
        """
        self.__var_pred_train = new_value
        
    @property
    def var_pred_test(self):
        """Valor del atributo var_pred_test.

        Retorna
        -------
        El valor del atributo var_pred_test.

        """
        return self.__var_pred_test
    
    @var_pred_test.setter
    def var_pred_test(self, new_value):
        """Cambia el valor del atributo var_pred_test.
        
        Parámetros
        ----------
            new_value: pandas.DataFrame
                El nuevo valor del atributo var_pred_test.
        """
        self.__var_pred_test = new_value
    

# referencias
# https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning
# https://www.kaggle.com/code/prashant111/comprehensive-guide-to-ann-with-keras/notebook