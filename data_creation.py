import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from dataLoaders import DigitData
from config import *
from funcs import get_dict_from_class, updateMetricsSheet, DataCreation
from models import FeatureExtractor
from itertools import permutations,combinations
from tqdm import tqdm
from torch.utils.data import DataLoader

looper=0
pixel = ['pixel' + str(i) for i in range(784)]


data_load = DigitData(**get_dict_from_class(DataLoad1))
data = data_load.data
#for i in range(784):
data[pixel] = 0
dataCreation = DataCreation(data_path=dataPath, image_path_=image_path)
# dataCreation.circles_and_rectngles(data)
data = pd.read_csv(str(dataPath) + '/train.csv')
dataCreation.shifter( data, data_count=10000, size=(112, 112),size2=(28,28))
# dataCreation = DataCreation(data_path='/home/pooja/PycharmProjects/digitRecognizer/rough/scale/data', image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/scale/images')
# data=dataCreation.scaler( data, data_count=10000, size=(112, 112),size2=(28,28), scales=4)
# dataCreation = DataCreation(data_path='/home/pooja/PycharmProjects/digitRecognizer/rough/shiftScale/data', image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/shiftScale/images')
# dataCreation.shifter( data, data_count=10000, size=(112*4, 112*4),size2=(28*4,28*4))
# #darker(data)





