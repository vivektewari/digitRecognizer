from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

from commonFuncs import packing
maxrows = 500
class DigitData(Dataset):
    def __init__(self, data_frame=None, label=None, pixel_col=None, reshape_pixel=None, path=None):
        if data_frame is None:
            if maxrows is None:
                data_frame = pd.read_csv(path, nrows=maxrows)
            else:
                data_frame = pd.read_csv(path,nrows=maxrows)
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

class DigitData_l(DigitData):
    def __init__(self, data_frame=None, label=None, pixel_col=None,localization_col=None, reshape_pixel=None, path=None):
        super().__init__(data_frame,label, pixel_col, reshape_pixel, path)
        self.localization_col = localization_col
    def __getitem__(self, idx):

        out = super().__getitem__(idx)
        out['targets'] =torch.tensor([out['targets'].tolist()]+packing.unpack(self.data.loc[idx][self.localization_col]), dtype=torch.float32)
        return out
class DigitData_mult_object(DigitData):
    def __init__(self, data_frame=None,data_frame_loc=None, label=None, pixel_col=None, localization_col=None, reshape_pixel=None,
                 path=None,path_loc=None):
        super().__init__(data_frame, label, pixel_col, reshape_pixel, path)
        if data_frame_loc is None:
            if maxrows is None:
                data_frame_loc = pd.read_csv(path_loc, nrows=maxrows)
            else:
                data_frame_loc = pd.read_csv(path_loc,nrows=maxrows)
        self.data_loc=data_frame_loc
        self.localization_col = localization_col
    def __getitem__(self, idx):
        """

        :param idx: index for which data to be return
        :return: target as tensor with
        """
        box=[]
        out = super().__getitem__(idx)
        data=self.data_loc[self.data_loc['index']==idx][self.labelCol+self.localization_col]

        for row in data.rows():
            box.append(torch.tensor(row))
        out['targets']=box

        return out


if __name__ == "__main__":
    from funcs import get_dict_from_class
    from config import DataLoad1_l

    class test_DigitData_l():
        def __init__(self):
            self.obj=DigitData_l(**get_dict_from_class(DataLoad1_l))
        def test_(self):

            return self.obj.__getitem__(1)
    c = test_DigitData_l()
    d = c.test_()
    stop = 1