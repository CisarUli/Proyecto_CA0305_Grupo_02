from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ActivityRegularization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class BPANN:
    def __init__(self, inputs, pred):
        """Constructor por defecto de la clase.

        Parámetros
        ----------
        ruta_archivo:str
            La ruta al archivo con los datos.

        Retorna
        -------
        
        """
        self.__X_train, self.__X_test, self.__var_pred_train, self.__var_pred_test = train_test_split(
            inputs, pred, test_size = 0.1, stratify = pred, random_state = 42
        )
        self.__modelo = None
        
    def __str__(self):
        """Descripción del objeto.
        
        Parámetros
        ----------

        Retorna
        -------
            Una descripción del objeto.
        """
        
        return f"Variable a predecir: Churn\nCon {self.__inputs}\
                \nDatos usados para entrenamiento y prueba:\nEntrenamiento\n{self.__X_train}\n{self.__var_pred_train}\n\
                    Prueba:\n{self.__X_test}\n{self.__var_pred_test}"

    def definir_modelo(self):
        """Define el modelo que se va a utilizar, en este caso una red neuronal con retropropagación.

        Parámetros
        ----------

        Retorna
        -------
        
        """
        input_dim = self.__X_train.shape[1]
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
                                   patience = 150,
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
        self.definir_modelo()
        self.entrenar()
        self.evaluar()
    
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
                
        Retorna
        -------
        
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
                
        Retorna
        -------
        
        """
        self.__var_pred_train = new_value
        
    @property
    def var_pred_test(self):
        """Valor del atributo var_pred_test.
        
        Parámetros
        ----------
        
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
        Retorna
        -------
        
        """
        self.__var_pred_test = new_value
    

# referencias
# https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning
# https://www.kaggle.com/code/prashant111/comprehensive-guide-to-ann-with-keras/notebook