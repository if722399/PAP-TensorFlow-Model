import pandas as pd

def subsampling(df,target_variable):
    # Obtener el valor de la clase que tiene una menor proporción
    min_sample_size = df.groupby(target_variable).precio_unitario.count().min()

    # Guardar en un diccionario las categorías de nuestra variable objetivo asi como el conteo de cada una 
    balance_dict = df.groupby(target_variable).precio_unitario.count().to_dict()

    # Realizar el balanceo (solo funciona para 2 categorías)
    for key in balance_dict:
        if balance_dict[key] != min_sample_size:
            max_class_sample = df[df[target_variable] == key].sample(min_sample_size)
            print(f'clase {key} balanceada')
        if balance_dict[key] == min_sample_size:
            min_class_sample = df[df[target_variable] == key]

    return pd.concat([max_class_sample,min_class_sample])
        