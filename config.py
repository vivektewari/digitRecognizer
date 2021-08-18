import time
from pathlib import Path
import random
import os
from dataLoaders import *
from losses import *
from models import *
from param_options import *

#from funcs import *
root = Path('/home/pooja/PycharmProjects/digitRecognizer/rough/localization')
image_path = root / 'images'
dataPath = root / 'data'
dataCreated = root / 'data' / 'dataCreated'
metricSheetPath = root / 'metricSheet2.csv'
devData = root / 'data' / 'dataCreated' / 'dev.csv'
holdData = root / 'data' / 'dataCreated' / 'holdout.csv'
saveDirectory = root / 'outputs'
device = 'cpu'
config_id = str(os.getcwd()).split()[-1]
startTime = time.time()
test_data = dataPath / 'test.csv'
lr = 0.05
epoch = 200

random.seed(23)
data_loader_param =DataLoad1_l_112 #DataLoad1
data_loader_param.path= devData
data_loader = DigitData_l #DigitData
loss_func =MultiBoxLoss(pixel_shape=(112,112),n_class=10)  #BCELoss
model_param =  ModelSSD
model =FTWithLocalization_prior
pre_trained_model ="/home/pooja/PycharmProjects/digitRecognizer/fold0/checkpoints/train.42.pth"
pre_trained_model =None #"/home/pooja/Downloads/train.298.pth"
#'/home/pooja/PycharmProjects/digitRecognizer/rough/localization/fold0/checkpoints/train.17.pth'




