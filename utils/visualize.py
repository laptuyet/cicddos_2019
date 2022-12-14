import matplotlib.pyplot as plt
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

IMG_DIR = os.path.join(os.path.abspath('.'), "images")


def save_confuse_matrix(y_true, y_pred, labels):
    confuse_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig = plt.figure(figsize=(16, 9))
    _ = sns.heatmap(confuse_matrix, annot=True, cmap="Blues", fmt="", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'confuse_matrix.pdf'))
