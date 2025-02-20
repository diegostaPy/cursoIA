import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generamos un conjunto de datos de prueba con valores atípicos
np.random.seed(0)
data = {
    'A': np.random.normal(loc=0, scale=1, size=100),
    'B': np.random.normal(loc=0, scale=1, size=100),
    'C': np.random.normal(loc=0, scale=1, size=100)
}

# Introducimos valores atípicos en la columna 'A'
data['A'][0] = 10
data['A'][1] = -10

df = pd.DataFrame(data)

# Creamos un boxplot para visualizar la distribución de los datos
df.boxplot()
plt.title('Boxplot de los datos')
plt.show()

# Calculamos el rango intercuartílico (IQR) para cada columna
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Definimos el límite para considerar valores atípicos
outlier_threshold = 1.5

# Identificamos valores atípicos utilizando el criterio del IQR
outliers = (df < (Q1 - outlier_threshold * IQR)) | (df > (Q3 + outlier_threshold * IQR))

print("Valores atípicos detectados:")
print(outliers.sum())
