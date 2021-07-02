import pandas as pd
import random
from dataLoaders import DigitData
from config import *
from funcs import get_dict_from_class, updateMetricsSheet
from models import FeatureExtractor
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

random.seed(24)
s = get_dict_from_class(DataLoad1)
data_load = DigitData(**get_dict_from_class(DataLoad1))
dev_dict = get_dict_from_class(DataLoad1)
hold_dict = get_dict_from_class(DataLoad1)
hold_dict['path'] = holdData

model = FeatureExtractor(**get_dict_from_class(Model1))
checkpoint = torch.load(saveDirectory / 'featureExtr_4_40.pth')
model.load_state_dict(checkpoint, strict=True)
model.eval()
collects = []
looper = 0
data = data_load.data['index']
for conf in [dev_dict, hold_dict]:
    preds_ = []
    actuals_ = []
    d = DataLoader(DigitData(data_frame=None, **conf),
                   batch_size=512,
                   shuffle=False,
                   num_workers=4,
                   pin_memory=True,
                   drop_last=False)

    for dict_ in tqdm(d):
        with torch.no_grad():
            predicted = model(dict_['image_pixels'])

            preds_.extend(torch.argmax(predicted, dim=1))
            actuals_.extend(dict_['targets'])

    temp = pd.DataFrame({'actual': [actuals_[i].item() for i in range(len(actuals_))],
                         'pred': [preds_[i].item() for i in range(len(preds_))]})

    data_load = DigitData(data_frame=None, **conf)
    temp['index'] = data_load.data['index']

    temp.to_csv(str(dataCreated) + '/' + str(looper) + '.csv')
    collects.append(actuals_)
    collects.append(preds_)
    looper += 1
# updateMetricsSheet(*collects, modelName='featureExtr_1', force=False)
