import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from dataLoaders import DigitData
from config import *
from funcs import get_dict_from_class, updateMetricsSheet
from models import FeatureExtractor
from itertools import permutations,combinations
from tqdm import tqdm
from torch.utils.data import DataLoader
pixel=['pixel'+str(i) for i in range(784)]
looper=0
def darker(data):
    global looper
    for i in range(data.shape[0]):
        labe = data.iloc[i].label
        pix_start = labe*28*2
        start = random.randint(0, 54)
        finish = random.randint(1, 55-start)+start

        data.iloc[i][['pixel'+str(j) for j in range(start+pix_start,finish+pix_start)]] = 255
        ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
        fig = plt.figure()
        plt.imshow(ar)
        fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/rough/images/' + str(i) +'_'+str(labe)+ '.png')
        plt.close(fig)
        looper+=1
    data.to_csv('/home/pooja/PycharmProjects/digitRecognizer/rough/newData.csv')

def circles_and_rectngles(data):
    length=8
    for i in range(1000):
        startRow = random.randint(0, 28-length-1)
        startCol = random.randint(0, 28-length-1)
        shp=random.choice([0, 1])
        data.iloc[i]['label'] = shp
        if shp:
            pixs = ['pixel'+str(startRow*28+startCol+k[0]*28+k[1]) for k in combinations([length-1-i for i in range(length)], 2)]
        else :
            pixs = ['pixel' + str(startRow * 28 + startCol + k[0] * 28 + k[1]) for k in combinations([i for i in range(length)], 2)]
        data.iloc[i][pixs] = 255
        ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
        fig = plt.figure()
        plt.imshow(ar)
        fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/rough/images/' + str(i)+"_"+str(shp) + '.png')
        plt.close(fig)

    data[0:1000].to_csv('/home/pooja/PycharmProjects/digitRecognizer/rough/newData.csv')
data_load = DigitData(**get_dict_from_class(DataLoad1))
data = data_load.data
#for i in range(784):
data[pixel] = 0
circles_and_rectngles(data)

#darker(data)





