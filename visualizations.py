# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.style.use('ggplot')


def balancing(df,target_variable='outlier'):
    conteo = df.groupby(target_variable).precio_unitario.count()
    plt.figure(figsize=(7,6))
    plt.bar(df[target_variable].unique(),conteo.values)
    plt.title('Balanceo de nuestra variable objetivo')
    plt.grid(alpha=.4)
    print(f'La clase de NO outliers tiene {conteo[0]}')
    print(f'La clase de Outliers tiene {conteo[1]}')


def evaluation_metrics(history,n_epochs):
    plt.plot(
    np.arange(1, n_epochs+1), 
    history.history['loss'], label='Loss'
    )
    plt.plot(
        np.arange(1,  n_epochs+1), 
        history.history['accuracy'], label='Accuracy'
    )
    plt.plot(
        np.arange(1,  n_epochs+1), 
        history.history['precision'], label='Precision'
    )
    plt.plot(
        np.arange(1,  n_epochs+1), 
        history.history['recall'], label='Recall'
    )
    plt.plot(
        np.arange(1,  n_epochs+1), 
        history.history['auc'], label='AUC'
    )
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend();


def plot_cm(labels, predictions,threshold):
  cm = confusion_matrix(labels, predictions)
  plt.figure(figsize=(9,7))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(threshold))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate NO_Outliers Detected (True Negatives): ', cm[0][0])
  print('Legitimate NO_Outliers Incorrectly Detected (False Positives): ', cm[0][1])
  print('Outliers Missed (False Negatives): ', cm[1][0])
  print('Outliers Detected (True Positives): ', cm[1][1])
  print('Total Outliers: ', np.sum(cm[1]))
  print('Total No_Outliers: ', np.sum(cm[0]))