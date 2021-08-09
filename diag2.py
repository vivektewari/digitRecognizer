from dataLoaders import DigitData
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import *
from funcs import get_dict_from_class
from models import FeatureExtractor, FTWithLocalization
from losses import BCELoss
from model_configs import Model0
from torch.utils.data import DataLoader
import pandas as pd
root='/home/pooja/PycharmProjects/digitRecon2/'+'rough/scale/'
input_image_size = [(28 * 4, 28 * 4)]
DataLoad1.reshape_pixel = input_image_size [0]
DataLoad1.path =Path(root) /'data' / 'dataCreated' / 'holdout.csv'
DataLoad1.pixel_col = ['pixel' + str(i) for i in
                              range(DataLoad1.reshape_pixel[0] * DataLoad1.reshape_pixel[1])]
s = get_dict_from_class(DataLoad1)
dev_dict = get_dict_from_class(DataLoad1)

data_load = DigitData(**get_dict_from_class(DataLoad1))
indexes=list(data_load.data['index'])
criterion = BCELoss()
model = Model0
model.input_image_dim= input_image_size [0]
model = FeatureExtractor(**get_dict_from_class(model))

#model = FCLayered(**get_dict_from_class(Model1))
if True:
    checkpoint = torch.load(str(saveDirectory)  +'/featureExtr_4_8.pth')


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
                   batch_size=500,
                   shuffle=False,
                   num_workers=1,
                   pin_memory=True,
                   drop_last=False)
# min max finder
    for dict_ in tqdm(d):
        with torch.no_grad():
            model.num_blocks=1
            data0 = model.cnn_feature_extractor(dict_['image_pixels']/255)
            model.num_blocks = 2
            data1 = model.cnn_feature_extractor(dict_['image_pixels']/255)
            x = data1.flatten(start_dim=1, end_dim=-1)
            x = model.activation_l(x)
            data2 = model.activation_l(model.fc1(x))
            min_max = [[torch.min(data0),torch.max(data0)], [torch.min(data1),torch.max(data1)],[torch.min(data2),torch.max(data2)]]
            break
    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    data = np.array([min_max[0][0] +i * (min_max[0][1] - min_max[0][0]) / 8 for i in range(9)])
    plt.imshow(data.reshape((3, 3)), aspect='auto')
    plt.title(str(min_max[0][0]) + "_" + str(min_max[0][1]))
    fig.add_subplot(1, 3,2)
    data = np.array([min_max[1][0] +i * (min_max[1][1] - min_max[1][0]) / 8 for i in range(9)])
    plt.imshow(data.reshape((3, 3)), aspect='auto')
    plt.title(str(min_max[1][0]) + "_" + str(min_max[1][1]))
    fig.add_subplot(1, 3, 3)
    data = np.array([min_max[2][0] + i * (min_max[1][1] - min_max[2][0]) / 8 for i in range(9)])
    plt.imshow(data.reshape((3, 3)), aspect='auto')
    plt.title(str(min_max[2][0]) + "_" + str(min_max[2][1]))
    fig.savefig(root+'/weightDist/layer_images/0_reference.png')
    plt.close(fig)


    d = DataLoader(DigitData(data_frame=None, **conf),
                   batch_size=1,
                   shuffle=False,
                   num_workers=1,
                   pin_memory=True,
                   drop_last=False)

    i = 0
    model.num_blocks=2
    for dict_ in tqdm(d):
        with torch.no_grad():
            index = indexes[i]
            label = dict_['targets'].item()
            for k in range(2):
                model.num_blocks=k+1
                data = model.cnn_feature_extractor(dict_['image_pixels']/255)#.flatten()
                channel=data.shape[-3]
                data=data.reshape((data.shape[-3],data.shape[-2],data.shape[-1]))

                fig = plt.figure()
                for i1 in range(channel):
                    fig.add_subplot(int(channel/5)+1, 5, i1 + 1)
                    plt.imshow(torch.reshape(data[i1], shape=(data.shape[-2],data.shape[-1])), vmin=min_max[0][0], vmax=min_max[0][1])

                fig.savefig(root+'/weightDist/layer_images/out_'+ str(
                    index) +'_'+str(k)+ "_" + str(label) + '.png', bbox_inches='tight')
                plt.close(fig)
            model.num_blocks = 2
            for k in range(2):
                if k ==0:
                    x = model.cnn_feature_extractor(dict_['image_pixels']/255)
                    x = x.flatten(start_dim=1, end_dim=-1)
                    x = model.activation_l(x)
                    data =model.activation_l(model.fc1(x))
                elif k ==1:
                    data=model(dict_['image_pixels'])
                fig = plt.figure()

                if k==0:plt.imshow(torch.reshape(data.flatten(), shape=(32, 8)),vmin=min_max[2][0],vmax=min_max[2][1])
                else:plt.imshow(torch.reshape(data.flatten(), shape=(1, 10)), vmin=0,
                               vmax=1)

                fig.savefig(root + '/weightDist/layer_images/out_'+ str(
                    index)+ '_'  + str(k+2)  + "_" + str(label) + '.png', bbox_inches='tight')
                plt.close(fig)
            i += 1
            if i == 100:
                break