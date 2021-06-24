import torch.nn as nn
import torch
EPSILON_FP16 = 1e-5
class BCELoss(nn.Module):
    def __init__(self):

        super().__init__()
        self.func = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):
        # bs, s, o = pred.shape
        # pred = pred.reshape(bs*s, o)
        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        return self.func(pred, actual)#, {}