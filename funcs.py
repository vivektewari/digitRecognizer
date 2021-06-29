from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def toImage(savePath):
    train = pd.read_csv(dataPath / 'train.csv',nrows=1000)
    plt.gray()
    for row in range(train.shape[0]):
        all = list(train.iloc[row])
        label = all[0]
        data = np.array(all[1:])
        #data=np.where(data>100,255,0) # add step to preprocess the data
        data = data.reshape((28, 28))
        fig = plt.figure()
        plt.imshow(data)
        fig.savefig(savePath / str(str(row) + "_" + str(label) + '.png'))
        plt.close(fig)
        if row % 1000 == 0: gc.collect()
    # print("time taken ".format(time.time()-startTime))


def getMetrics(actual, predicted):
    """
    :param actual: actual series
    :param predicted: predicted series
    :return: list of accuracy ,precision ,recall and f1
    """
    met = []
    metrics = [accuracy_score, precision_score, recall_score, f1_score]
    for m in metrics:
        if m == accuracy_score:
            met.append(m(actual, predicted))
        else:
            met.append(m(actual, predicted, average="macro"))
    return met


def updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc=metricSheetPath, modelName="", extraInfo="",
                       force=False):
    model = 'model'
    f = pd.read_csv(loc, index_col='index')
    if modelName in list(f[model]):
        if not force: raise Exception("model exist. try with Force as True or different model name")
        # else:f.drop(f[f[model]==modelName].index,axis=0)
    metricsDev, metricsHold = getMetrics(dev_actual, dev_pred), getMetrics(hold_actual, hold_pred)
    entryVal = modelName, *metricsDev, *metricsHold, extraInfo
    dict = {}

    for i in range(f.shape[1]):
        dict[f.columns[i]] = entryVal[i]

    pd.DataFrame(dict, index=[f.shape[0]]).to_csv(loc, mode='a', header=False)


def get_dict_from_class(class1):
    return {k: v for k, v in class1.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}
