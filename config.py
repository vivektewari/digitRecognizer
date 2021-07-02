import time
from pathlib import Path
import random
root = Path('/home/pooja/PycharmProjects/digitRecognizer/rough/shift')
image_path = root / 'images'
dataPath = root / 'data'
dataCreated = root / 'data' / 'dataCreated'
metricSheetPath = root / 'metricSheet.csv'
devData = root / 'data' / 'dataCreated' / 'dev.csv'
holdData = root / 'data' / 'dataCreated' / 'holdout.csv'
saveDirectory = root / 'outputs'
device = 'cpu'
startTime = time.time()
test_data = dataPath / 'test.csv'
lr = 0.05
epoch = 40
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




class DataLoad1:
    # data_frame = None
    label = 'label'
    pixel_col = ['pixel' + str(i) for i in range(28 * 28)]
    reshape_pixel = (28, 28)
    path = devData
