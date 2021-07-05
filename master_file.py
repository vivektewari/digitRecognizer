import os, importlib, sys

import pandas as pd


from config import *
import config, train, inference , start
# temp = pd.read_csv(metricSheetPath,index_col='index')
# temp = temp.drop(temp.index)
# temp.to_csv(metricSheetPath)

def iterator(directory):
    os.chdir(directory)
    importlib.reload(sys.modules['config'])
    # importlib.reload(sys.modules['start']) # only needed first time
    importlib.reload(sys.modules['train'])
    train.train(config.Model1,config.DataLoad1)
    #importlib.reload(sys.modules['inference'])
    inference.get_inference(str(2)+directory)



base = '/home/pooja/PycharmProjects/digitRecognizer/'
dirs=['','rough/shift/','rough/scale/','rough/shiftScale/']
dirs=['rough/shift/']

for d in dirs:
    iterator(base+d)
