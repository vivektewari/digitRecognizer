import os, importlib, sys

import pandas as pd

import config, train, inference, start
from config import *
temp = pd.read_csv(metricSheetPath,index_col='index')
temp = temp.drop(temp.index)
temp.to_csv(metricSheetPath)

def iterator(directory):
    os.chdir(directory)
    importlib.reload(sys.modules['config'])
    # importlib.reload(sys.modules['start']) #only needed first time
    importlib.reload(sys.modules['train'])
    importlib.reload(sys.modules['inference'])
    inference.get_inference(directory)



base = '/home/pooja/PycharmProjects/digitRecognizer/'
dirs=['','rough/shift/','rough/scale/','rough/shiftScale/']

for d in dirs:
    iterator(base+d)
