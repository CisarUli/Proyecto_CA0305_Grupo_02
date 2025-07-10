from BPANN import BPANN
from LimpiezaDatos import LimpiezaDatos
from NB import NB

datos_limpios = LimpiezaDatos("Telco-Customer-Churn4.csv")
datos_limpios.limpiar()

bpan = BPANN(datos_limpios.datos[0], datos_limpios.datos[1])
nb = NB(datos_limpios.datos[0], datos_limpios.datos[1])

bpan.probar()
nb.probar()