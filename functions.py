import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def subsampling(df, target_variable):
    # Obtener el valor de la clase que tiene una menor proporción
    min_sample_size = df.groupby(target_variable).precio_unitario.count().min()

    # Guardar en un diccionario las categorías de nuestra variable objetivo asi como el conteo de cada una
    balance_dict = df.groupby(
        target_variable).precio_unitario.count().to_dict()

    # Realizar el balanceo (solo funciona para 2 categorías)
    for key in balance_dict:
        if balance_dict[key] != min_sample_size:
            max_class_sample = df[df[target_variable]
                                  == key].sample(min_sample_size)
            print(f'clase {key} balanceada')
        if balance_dict[key] == min_sample_size:
            min_class_sample = df[df[target_variable] == key]

    return pd.concat([max_class_sample, min_class_sample])


def label_encoder(data: "DataFrame"):
    for col in data.columns:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col].fillna("None", inplace=True)
            le.fit(list(data[col].astype(str).values))
            data[col] = le.transform(list(data[col].astype(str).values))
        else:
            data[col].fillna(-999, inplace=True)
    return data

def decile_analysis(predictions: "Array", labels: "Array"):

    test_pred = np.array([predictions[i][0] for i in range(len(predictions))])

    test_results = pd.DataFrame(
    data = 
    {
        "predictions": test_pred,
        "label": labels
    }
    )

    test_results = test_results.sort_values(by=['predictions'], ascending=False)
    test_results.reset_index(inplace=True)

    deciles = np.array_split(np.array(test_results["predictions"]), 10)
    dec_labels = np.array_split(np.array(test_results["label"]), 10)

    decile_analysis = pd.DataFrame(
    data = {
        "Decile": np.arange(1, 11),
        "Batch": [len(deciles[i]) for i in range(len(deciles))],
        "Cumulative Batch": np.cumsum([len(deciles[i]) for i in range(len(deciles))]),
        "Cumulative % Batch": np.round(np.cumsum([len(deciles[i]) for i in range(len(deciles))])/len(test_pred), 4),
        "True label": [sum(dec_labels[i]) for i in range(len(dec_labels))],
        "True label %": np.round([sum(dec_labels[i])/sum(test_results["label"]) for i in range(len(dec_labels))],4),
        "Cumulative label %": np.round(np.cumsum([sum(dec_labels[i])/sum(test_results["label"]) for i in range(len(dec_labels))]), 4),
        "Probability Range" : [str(np.round(deciles[i].max(),4)) + " - " + str(np.round(deciles[i].min(),4)) for i in range(len(deciles))]   
    }
    )
    
    decile_analysis.set_index('Decile', inplace =  True)
    return decile_analysis
