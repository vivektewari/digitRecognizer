import time
from pathlib import Path
import random
import os
from dataLoaders import *
from losses import *
from models import *
from param_options import *
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
lr = 0.1
epoch = 20
maxrows = 1000
random.seed(23)
data_loader_param =DataLoad1_l #DataLoad1
data_loader_param.path= devData
data_loader = DigitData_l #DigitData
loss_func =LocalizatioLoss  #BCELoss
model_param = Model3
model =FeatureExtractor