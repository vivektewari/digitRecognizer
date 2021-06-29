from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np

class DigitData(Dataset):
    def __init__(self, data_frame=None, label=None, pixel_col=None, reshape_pixel=None, path=None):
        if data_frame is None: data_frame = pd.read_csv(path,nrows=10000)  # ,nrows=1000
        self.data = data_frame
        self.data.reset_index(inplace=True, drop=True)
        self.labelCol = label
        self.pixelCol = pixel_col
        self.reshape_pixel = reshape_pixel

    def __getitem__(self, idx):
        data = self.data.loc[idx]
        label, pixel = torch.tensor(data[self.labelCol], dtype=torch.long), torch.tensor(data[self.pixelCol]
                                                                                         , dtype=torch.float32)
        pixel = pixel.reshape(self.reshape_pixel)
        pixel = torch.where(pixel > torch.tensor(100), torch.tensor(255.0), torch.tensor(0.0))  # add step to preprocess the data
        pixel = torch.unsqueeze(pixel, dim=0)
        return {'targets': label, 'image_pixels': pixel}

    def __len__(self):
        return self.data.shape[0]
