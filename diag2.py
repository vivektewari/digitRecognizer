from dataLoaders import DigitData
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import *
from funcs import get_dict_from_class
from models import FeatureExtractor, FTWithLocalization
from losses import BCELoss

from torch.utils.data import DataLoader
import pandas as pd

s = get_dict_from_class(DataLoad1)
dev_dict = get_dict_from_class(data_loader_param)
data_load = data_loader(**get_dict_from_class(DataLoad1))
indexes=list(data_load.data['index'])
criterion = BCELoss()
model = model(**get_dict_from_class(Model1))
#model = FCLayered(**get_dict_from_class(Model1))
if True:
    checkpoint = torch.load(pre_trained_model)['model_state_dict']
    model.load_state_dict(checkpoint)
    model.eval()

# model = FeatureExtractor(**get_dict_from_class(Model1))
# model = FCLayered(**get_dict_from_class(Model1))
for conf in [dev_dict]:
    preds_ = []
    actuals_ = []


    channel = 10
    min_max=[[0,0],[0,0]]
    d = DataLoader(DigitData(data_frame=None, **conf),
                   batch_size=1000,
                   shuffle=False,
                   num_workers=1,
                   pin_memory=True,
                   drop_last=False)
# min max finder
    for dict_ in tqdm(d):
        with torch.no_grad():
            model.num_blocks = 1
            data0 = model(dict_['image_pixels'])
            model.num_blocks = 2
            data1 = model(dict_['image_pixels'])
            min_max = [torch.min(data0),torch.max(data0)], [torch.min(data1),torch.max(data1)]
            break
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    data = np.array([min_max[0][0] +i * (min_max[0][1] - min_max[0][0]) / 8 for i in range(9)])
    plt.imshow(data.reshape((3, 3)), aspect='auto')
    plt.title(str(min_max[0][0]) + "_" + str(min_max[0][1]))
    fig.add_subplot(1, 2,2)
    data = np.array([min_max[1][0] +i * (min_max[1][1] - min_max[1][0]) / 8 for i in range(9)])
    plt.imshow(data.reshape((3, 3)), aspect='auto')
    plt.title(str(min_max[1][0]) + "_" + str(min_max[1][1]))
    fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/layer_1mages/0_reference.png')
    plt.close(fig)


    d = DataLoader(DigitData(data_frame=None, **conf),
                   batch_size=1,
                   shuffle=False,
                   num_workers=1,
                   pin_memory=True,
                   drop_last=False)

    i = 0
    model.num_blocks=1
    for dict_ in tqdm(d):
        with torch.no_grad():

            label = dict_['targets'].item()

            data = model(dict_['image_pixels']).flatten()

            data=data.reshape(channel,49)
            index=indexes[i]
            fig = plt.figure(figsize=(2, 5))
            for i1 in range(channel):
                fig.add_subplot(4, 3, i1 + 1)
                plt.imshow(torch.reshape(data[i1], shape=(7, 7)), vmin=min_max[0][0], vmax=min_max[0][1])

            fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/layer_1mages/1out_' + str(
                index) + "_" + str(label) + '.png', bbox_inches='tight')
            plt.close(fig)
            i += 1
            if i == 100 :
                break
    i = 0
    model.num_blocks = 2
    channel = 10

    for dict_ in tqdm(d):
                with torch.no_grad():

                    label = dict_['targets'].item()
                    data = model(dict_['image_pixels']).flatten()
                    data = data.reshape(channel, 1)
                    index = indexes[i]
                    fig = plt.figure()
                    # for i1 in range(channel):
                    #     fig.add_subplot(10, 1, i1 + 1)
                    plt.imshow(data,vmin=min_max[1][0], vmax=min_max[1][1])

                    fig.savefig('/home/pooja/PycharmProjects/digitRecognizer/weightDist/layer_1mages/2out_' + str(
                        index) + "_" + str(label) + '.png')
                    plt.close(fig)

                    i += 1
                    if i == 100:
                        break
            # plt.imshow(torch.reshape(dict_['image_pixels'],shape=(28,28)))
            # plt.show()
            # c = 0
