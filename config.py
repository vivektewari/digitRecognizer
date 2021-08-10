import time
from pathlib import Path
import random
import os
from dataLoaders import *
from losses import *
from models import *
from param_options import *

#from funcs import *
root = Path(os.getcwd())#rough/shift
image_path = root / 'images'
dataPath = root / 'data'
dataCreated = root / 'data' / 'dataCreated'
metricSheetPath = root / 'metricSheet2.csv'
devData = root / 'data' / 'dataCreated' / 'dev.csv'
holdData = root / 'data' / 'dataCreated' / 'holdout.csv'
saveDirectory = root / 'outputs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_id = str(os.getcwd()).split()[-1]
startTime = time.time()
test_data = dataPath / 'test.csv'
lr =0.5
epoch = 1000

random.seed(23)
data_loader_param =DataLoad1_l_112 #DataLoad1
#data_loader_param.path= devData
data_loader = DigitData_l #DigitData
loss_func =LocalizatioLoss  #BCELoss
model_param = Model3_112_2
model =FTWithLocalization
pre_trained_model ="/train.29.pth"
#'/home/pooja/PycharmProjects/digitRecognizer/rough/localization/fold0/checkpoints/train.17.pth'




