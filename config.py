import time
from pathlib import Path

root = Path('/home/pooja/PycharmProjects/digitRecognizer')
dataPath = root / 'data'
dataCreated = root / 'data' / 'dataCreated'
metricSheetPath = root / 'metricSheet.csv'
devData = dataCreated / 'dev.csv'
holdData = dataCreated / 'holdout.csv'
saveDirectory = root / 'outputs'
device = 'cpu'
startTime = time.time()
test_data = dataPath / 'test.csv'
lr = 0.009


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
    pads = [1, 1]
    strides = [1, 1]
    pools = [3, 5]
    # layer 28,28, pa 30,30 c 28,28 p 14,14 ,pa 16,16 c 14,14 p 7,7 pa 7,7 c 5,5, p 1,1
    # layer 28,28, pa 28,28 c 21,21 p 7,7 ,pa 7,7 c 5,5 p 1,1
    #0.89 in valid
    channels = [16,30, 20]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [9 , 3,3]
    pads = [1, 1,1]
    strides = [1, 1,1]
    pools = [3, 3,2]
    #############################
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5, 3]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [1040, 256]#1040





class DataLoad1:
    # data_frame = None
    label = 'label'
    pixel_col = ['pixel' + str(i) for i in range(28 * 28)]
    reshape_pixel = (28, 28)
    path = devData
