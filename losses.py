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
        #pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        return self.func(pred, actual)

class LocalizatioLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bounding_loss = self.bounding_box_l2
        self.classification_loss =self.l2_loss# BCELoss()#
        self.bouding_conv_func= lambda x :x
    def l2_loss(self,pred, actual):
        #actual=torch.where(actual==1,1 ,0)
        loss =(torch.pow(torch.mean(torch.pow(1-pred[[i for i in range(len(pred))],[actual.tolist()]],2),dim=1),1/2))
        return loss[0]
    def bounding_box_l2(self,pred, actual):
        """

        :param pred:list of tuple in n dimension
        :param actual: list of tuple in n dimension
        :return: sum of l2 distance for each point
        """
        num_points = len(pred)
        loss = 0#torch.zeros(pred[0].shape)
        for i in range(num_points):
            loss +=torch.sum(torch.pow(torch.sum(torch.pow(pred[i]-actual[i],2),dim=1),1/2))

        return loss/(pred[0].shape[0]*2)

    def loss_calc(self, pred_classif, actual_classif,pred_bounding,actual_bounding):
        loss=self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))+10*self.classification_loss(pred_classif, actual_classif)
        return loss
    def convert(self,x):
        x_len=x.shape[1]-4

        return x[:,0:x_len],[x[:,x_len:x_len+2],x[:,x_len+2:x_len+4]]
    def forward(self,pred,actual):
        pred_classif, pred_bounding=self.convert(pred)
        actual_classif, actual_bounding=self.convert(actual)
        return self.loss_calc( pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0],pred_bounding,actual_bounding)
    def get_individual_loss(self,actual,pred):
        pred_classif, pred_bounding = self.convert(pred)
        actual_classif, actual_bounding = self.convert(actual)
        boundig_loss = self.bouding_conv_func(self.bounding_loss(pred_bounding,actual_bounding))
        classification_loss=self.classification_loss(pred_classif, torch.tensor(actual_classif,dtype=torch.long)[:,0])
        return boundig_loss , classification_loss






