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

        return self.func(pred, actual)

class LocalizatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bounding_loss = self.bounding_box_l2
        self.classification_loss = nn.CrossEntropyLoss()
    def bounding_box_l2(self,pred, actual):
        """

        :param pred:list of tuple in n dimension
        :param actual: list of tuple in n dimension
        :return: sum of l2 distance for each point
        """
        num_points = len(pred)
        loss = torch.zeros(pred[0].shape)
        for i in range(num_points):
            loss +=torch.pow(pred[i],2)-torch.pow(actual[i],2)

        return sum(sum(loss))

    def loss_calc(self, pred_classif, actual_classif,pred_bounding,actual_bounding):
        loss=torch.sigmoid(self.bounding_loss(pred_bounding,actual_bounding))+10*self.classification_loss(pred_classif, actual_classif[:,0])
        return loss
    def convert(self,x):
        x_len=x.shape[1]-4
        return x[:,0:x_len],[x[:,x_len:x_len+2],x[:,x_len+2:x_len+4]]
    def forward(self,pred,actual):
        pred_classif, pred_bounding=self.convert(pred)
        actual_classif, actual_bounding=self.convert(actual)
        return self.loss_calc( pred_classif, torch.tensor(actual_classif,dtype=torch.long),pred_bounding,actual_bounding)




