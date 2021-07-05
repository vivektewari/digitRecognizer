from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, random, cv2
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch


def toImage(savePath):
    train = pd.read_csv(dataPath / 'train.csv', nrows=100)
    plt.gray()
    for row in range(train.shape[0]):
        all = list(train.iloc[row])
        label = all[0]
        data = np.array(all[1:])
        data = np.where(data > 100, 255, 0)  # add step to preprocess the data
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


class DataCreation:
    def __init__(self, data_path=dataPath, image_path_=None):
        self.data_path = data_path
        self.image_path = image_path_
        self.to_csv = True

    def darker(self, data):

        pixel = ['pixel' + str(i) for i in range(784)]
        for i in range(data.shape[0]):
            labe = data.iloc[i].label
            pix_start = labe * 28 * 2
            start = random.randint(0, 54)
            finish = random.randint(1, 55 - start) + start

            data.iloc[i][['pixel' + str(j) for j in range(start + pix_start, finish + pix_start)]] = 255
            ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
            fig = plt.figure()
            plt.imshow(ar)
            fig.savefig(str(image_path) + '/' + str(i) + '_' + str(labe) + '.png')
            plt.close(fig)

        data.to_csv(str(self.data_Path) + '/newData.csv')

    def circles_and_rectngles(self, data):
        length = 8
        pixel = ['pixel' + str(i) for i in range(784)]
        for i in range(1000):
            startRow = random.randint(0, 28 - length - 1)
            startCol = random.randint(0, 28 - length - 1)
            shp = random.choice([0, 1])
            data.iloc[i]['label'] = shp
            if shp:
                pixs = ['pixel' + str(startRow * 28 + startCol + k[0] * 28 + k[1]) for k in
                        combinations([length - 1 - i for i in range(length)], 2)]
            else:
                pixs = ['pixel' + str(startRow * 28 + startCol + k[0] * 28 + k[1]) for k in
                        combinations([i for i in range(length)], 2)]
            data.iloc[i][pixs] = 255

            ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
            fig = plt.figure()
            plt.imshow(ar)
            fig.savefig(str(image_path) + '/' + str(i) + "_" + str(shp) + '.png')
            plt.close(fig)

        data.to_csv(str(dataPath) + '/newData.csv')

    def get_fig(self, data, img_no, label):
        if self.image_path is not None:
            fig = plt.figure()
            plt.imshow(data.astype(int))
            fig.savefig(str(self.image_path) + '/' + str(img_no) + "_" + str(label) + '.png')
            plt.close(fig)

    def black_image(self, data_count=100, size=(112, 112), ret=True):
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        data = pd.DataFrame(columns=['label'] + pixel, index=[i for i in range(data_count)])
        for i in range(data_count):
            ar = np.zeros(shape=size, dtype=int)
            data.loc[i] = [0] + list(ar.flatten())
            if not ret:
                self.get_fig(ar, i, 0)
        if ret:
            return data
        else:
            data.to_csv(str(self.data_path) + '/newData.csv')

    def shifter(self, data, data_count=100, size=(112, 112),size2=(28,28)):
        data_big = self.black_image(data_count, size)
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        pixel2 = ['pixel' + str(i) for i in range(size2[0] * size2[1])]
        for i in range(data_count):
            temp = np.array(data_big.iloc[i][pixel]).reshape(size)

            start_row, start_col = random.choice(range(0, size[0] - size2[0] + 1)),  \
                                        random.choice(range(0, size[1] - size2[1] + 1))
            temp[start_row:start_row + size2[0], start_col:start_col + size2[1]] = \
                np.array(data.iloc[i][pixel2]).reshape(size2)
            data_big.iloc[i][pixel] = temp.flatten()
            data_big.iloc[i]['label'] = data.iloc[i]['label']
            self.get_fig(temp, i, data.iloc[i]['label'])
        if self.to_csv:
            data_big.to_csv(str(self.data_path) + '/newData.csv')
        return data_big

    def scaler(self, data, data_count=100, size=(112, 112),size2=(28,28),scales=4):
        data_big = self.black_image(data_count, size)
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        pixel2 = ['pixel' + str(i) for i in range(size2[0] * size2[1])]
        for i in range(data_count):
            temp = np.array(data_big.iloc[i][pixel]).reshape(size)
            scale=random.choice(range(1,scales+1))
            res = cv2.resize(np.array(data.iloc[i][pixel2]).reshape(size2).astype('float32')
                             , dsize=(size2[0]*scale, size2[1]*scale)
                             , interpolation=cv2.INTER_CUBIC)
            temp[0: size2[0]*scale, 0:size2[1]*scale] = np.array(res)
            data_big.iloc[i][pixel] = temp.flatten()
            data_big.iloc[i]['label'] = data.iloc[i]['label']
            self.get_fig(temp, i, data.iloc[i]['label'])
        if self.to_csv:
            data_big.to_csv(str(self.data_path) + '/newData.csv')
        return data_big
