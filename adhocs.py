import pandas as pd
import numpy as np
import torch

from funcs import clusterring,vison_utils
from config import devData,dataCreated
from commonFuncs import packing
localization_col = 'localisation'
file=pd.read_csv(devData,nrows=10000)[[localization_col]]

temp=np.array(list(file[localization_col ].apply(lambda x:packing.unpack(x))))
#temp=vison_utils.xy_to_cxcy(torch.Tensor(temp)/28)
temp=vison_utils.get_scale_aspect(xy=torch.tensor(temp/28))
temp=torch.stack(list(temp), dim=1)
clusterring.kmeans(temp,standardized=True) # selected 10 clusters
centroids = clusterring.kmeans(temp,n_cluster=5,standardized=True)
pd.DataFrame({'aspect':centroids[:,0],'scale':centroids[:,1]}).to_csv(str(dataCreated)+'//centroids.csv')



