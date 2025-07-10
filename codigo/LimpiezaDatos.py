from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class LimpiezaDatos:
    
    def __init__(self, ruta_archivo: str):
        """Constructor por defecto de la clase.

        Parámetros
        ----------
        ruta_archivo: str
            La ruta al archivo con los datos.

        Retorna
        -------
        
        """
        self.__ruta_archivo = ruta_archivo
        self.__datos = None
        
    def __str__(self):
        """Descripción del objeto.
        
        Parámetros
        ----------

        Retorna
        -------
            Una descripción del objeto.
        """
        return f"Limpiando archivo {self.__ruta_archivo}"
        
    
    def limpiar(self):
        """Prepara los datos para el modelo.
        
        Parámetros
        ----------

        Retorna
        -------
        
        """
        datos = pd.read_csv(self.__ruta_archivo)
        datos = pd.get_dummies(datos, columns = ['Contract', 'PaymentMethod'], drop_first = True)
        var_pred = datos['Churn'].astype(int)
        datos_input = datos.drop(columns = 'Churn')
        
        imputer = SimpleImputer(strategy = 'mean')
        datos_imputados = pd.DataFrame(imputer.fit_transform(datos_input), columns = datos_input.columns)
        numeric_cols = datos_imputados.select_dtypes(include = ['float64', 'int64']).columns
        scaler = MinMaxScaler()
        datos_imputados[numeric_cols] = scaler.fit_transform(datos_imputados[numeric_cols])        
        lasso = LassoCV(cv = 5).fit(datos_imputados, var_pred)
        selected_features = datos_imputados.columns[lasso.coef_ != 0]
        datos_input = datos_imputados[selected_features]
        
        ros = RandomOverSampler(random_state = 42)
        inputs_balanceados, preds_balanceados = ros.fit_resample(datos_input, var_pred)
        self.__datos = [inputs_balanceados, preds_balanceados]
    
    @property
    def ruta_archivo(self):
        """Valor del atributo ruta_archivo.
        
        Parámetros
        ----------
        
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
                
        Retorna
        -------
        
        """
        self.__ruta_archivo = new_value
        
    @property
    def datos(self):
        """Valor del atributo datos.
        
        Parámetros
        ----------

        Retorna
        -------
        El valor del atributo datos.

        """
        return self.__datos
    
    @datos.setter
    def datos(self, new_value):
        """Cambia el valor del atributo datos.
        
        Parámetros
        ----------
            new_value: str
                El nuevo valor del atributo datos.
                
        Retorna
        -------
        
        """
        self.__ruta_archivo = new_value 