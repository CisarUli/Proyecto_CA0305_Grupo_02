import sys
sys.path.append('codigo/')

from BPANN import BPANN
from LimpiezaDatos import LimpiezaDatos
from NB import NB
from XGBModel import XGBModel

datos_limpios = LimpiezaDatos("datos/Telco-Customer-Churn4.csv")
datos_limpios.limpiar()

bpan = BPANN(datos_limpios.datos[0], datos_limpios.datos[1])
nb = NB(datos_limpios.datos[0], datos_limpios.datos[1])
xgboost = XGBModel()

bpan.probar()
nb.probar()
xgboost.evaluar(datos_limpios.datos[0], datos_limpios.datos[1])
