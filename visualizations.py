# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
plt.style.use('ggplot')


def balancing(df, target_variable='outlier'):
    conteo = df.groupby(target_variable).precio_unitario.count()
    plt.figure(figsize=(7, 6))
    plt.bar(df[target_variable].unique(), conteo.values)
    plt.title('Balanceo de nuestra variable objetivo')
    plt.grid(alpha=.4)
    print(f'La clase de NO outliers tiene {conteo[0]}')
    print(f'La clase de Outliers tiene {conteo[1]}')


def evaluation_metrics(history, n_epochs):
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
    plt.legend()


def plot_cm(labels, predictions, threshold):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate NO_Outliers Detected (True Negatives): ', cm[0][0])
    print(
        'Legitimate NO_Outliers Incorrectly Detected (False Positives): ', cm[0][1])
    print('Outliers Missed (False Negatives): ', cm[1][0])
    print('Outliers Detected (True Positives): ', cm[1][1])
    print('Total Outliers: ', np.sum(cm[1]))
    print('Total No_Outliers: ', np.sum(cm[0]))


def balancing2(data, bar_chart=False):
    class_count = pd.value_counts(data['outlier'], sort=True).sort_index()
    fig = go.Figure(data=[go.Pie(labels=['non-outlier', 'outlier'],
                                 values=class_count,
                                 pull=[0, 0.1],
                                 opacity=0.85)])
    fig.update_layout(
        title_text="Price Classification in Outlier - Non-Outlier")
    fig.show()

    if bar_chart:
        fig = px.histogram(data, x="outlier",
                           title='Price Classification in Outlier - Non-Outlier',
                           opacity=0.85,  # represent bars with log scale
                           color='outlier', text_auto=True)

        fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })

        fig.show()
