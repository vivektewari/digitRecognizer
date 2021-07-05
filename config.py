import time
from pathlib import Path
import random
import os

root = Path(os.getcwd())#rough/shift
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
epoch = 1
maxrows = 1000
random.seed(23)
class Model1:
    # layer 28,28, pa 29,29 c 27,27 p 9,9 ,pa 10,10 c 5,5 p 1,1

    channels = [30, 10]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [3, 6]
    pads = [1, 1]
    strides = [1, 1]
    pools = [3, 5]
    # layer 28,28, pa 28,28 c 21,21 p 7,7 ,pa 7,7 c 5,5 p 1,1
    channels = [10, 10]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [8, 3]
    pads = [1, 0]
    strides = [1, 1]
    pools = [3, 5]
# best
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [256, 10]#1040
class Model2:
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [None, 10]#1040




class DataLoad1:
    # data_frame = None
    label = 'label'
    reshape_pixel = (28, 28)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path = devData


