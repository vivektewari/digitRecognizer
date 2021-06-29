from prettytable import PrettyTable
import torch

from funcs import get_dict_from_class
from models import FeatureExtractor
from config import *
model = FeatureExtractor(**get_dict_from_class(Model1))
checkpoint = torch.load(saveDirectory / 'featureExtr_4_1.pth')
model.load_state_dict(checkpoint, strict=True)
model.eval()
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


count_parameters(model)