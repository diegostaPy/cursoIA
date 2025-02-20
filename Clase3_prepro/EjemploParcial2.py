import pandas as pd
import numpy as np

# Función para calcular los coeficientes de regresión manualmente
def regresion_manual(X, y):
    # Agregar una columna de unos para el término independiente
    unos=np.ones_like(X)
    A=np.vstack((unos,X)).T
    print(np.shape(A))
    AtA=np.matmul(np.transpose(A),A)
    invAtA=np.linalg.inv(AtA)
    Atb=np.matmul(np.transpose(A),y)
    w=np.matmul(invAtA,Atb)
    # Calcular los coeficientes utilizando la fórmula de la pseudo inversa
    print(y)
    coeficientes =w

    return coeficientes  

# Función para predecir los valores de y
def predecir(X, coeficientes):
   # Xm = # completar con lo mismo de la linea 7
    
    return  Xm@coeficientes

# Calcular métricas de evaluación manualmente
def rmse(y_true, y_pred):
    #error=
    return np.sqrt(np.mean((error) ** 2))

def r2F(y_true, y_pred):
   # https://es.wikipedia.org/wiki/Coeficiente_de_determinaci%C3%B3n
   
#    numerador = ((completar).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominador)

# Función para ajustar el modelo y evaluarlo
def ajustar_evaluar_modelo(X, y):
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ =[]#completar
    rmse_val = []#completar
    return coeficientes, y_pred, r2_, rmse_val


opcion=int(input())
# Cargar los datos
data = pd.read_csv('Mediciones.csv')

# Definir las columnas de características (X) y la columna de objetivo (y)
if opcion==1:
    #imprimir numero de filas y numero de columnas
    print(data.shape)
    #seleccionar las caracteristicas(variables dependientes) y el objetivo
    caracteristicas =['VTI_F', 'PEEP', 'BPM', 'VTE_F'] #[completar]
    objetivo ='Pasos'#
    
    print(caracteristicas)
    print(objetivo)
elif opcion==2: 
    # modelo completo solo con VTI_F, completar la funcion regresion manual
    
    X = data['VTI_F'].values
    y = data['Pasos'].values
    coef =regresion_manual(X, y)
    print(coef)
elif opcion==3: 
    # modelo completo solo con VTI_F, completar las funciones que definen las métricas
    #X = data[completar]
    #y = data[completar]
    coef = regresion_manual(X, y)
    print( coef)
    y_pred = predecir(X,coef)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    # imprimir los primeros 2 elementos de y e y_pred
    #  print(y[:3],  y_pred [COMPLETAR])
    print(y[:3],  y_pred [COMPLETAR])
    # imprimir r2 y rmse
    print(r2_,  rmse_val )
elif opcion==4: 
    # modelo completo solo con VTI_F, completar la función ajustar_evaluar_modelo
    X_todo =[]  #data[completar]
    y =[] # data[completar]
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y)
    print(r2_todo, rmse_todo)
elif opcion==5:
   # Completar la combinaciones de características de los modelos solicitados 
    models = {
        'Modelo_1': ['VTI_F'],
        'Modelo_2': ['VTI_F', 'BPM']
      #COMPLETAR EL DICCIONARIO
    }
    for nombre_modelo, lista_caracteristicas in models.items():
        X = []#data[completar]
        y = data['Pasos']
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo,r2, rmse_val)
elif opcion==6:
    # Modelos para cada combinación de PEEP y BPM
    valores_peep_unicos = []#completar sugerencia, utilizar unique()
    valores_bpm_unicos = [] #completar
    print(valores_peep_unicos)
    print(valores_bpm_unicos)
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:
            
            
            
            datos_subset = data #completar el filtrado de datos, se deben filtrar los datos para cada para par de PEEP y BPM
            
            
            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print(peep, bpm, r2_subset, rmse_subset)
            predicciones_totales.append(y_pred_subset)
    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y=data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global', r2_global, rmse_global)