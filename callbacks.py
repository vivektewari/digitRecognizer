


import torch

from funcs import getMetrics,DataCreation
from utils.visualizer import Visualizer
import os,cv2

from catalyst.dl  import  Callback, CallbackOrder,Runner

class MetricsCallback(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "acc_pre_rec_f1",
                 ):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval

        self.visualizer = Visualizer()

    # def on_batch_end(self,state: State):# #
    #     targ = state.input[self.input_key].detach().cpu().numpy()
    #     out = state.output[self.output_key]
    #
    #     clipwise_output = out[self.model_output_key].detach().cpu().numpy()
    #
    #     self.prediction.append(clipwise_output)
    #     self.target.append(targ)
    #
    #     y_pred = clipwise_output.argmax(axis=1)
    #     y_true = targ.argmax(axis=1)

    # score = f1_score(y_true, y_pred, average="macro")
    # state.batch_metrics[self.prefix] = score

    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:
            preds = torch.argmax(state.batch['logits'], dim=1)
            print("{} is {}".format(self.prefix, getMetrics(state.batch['targets'], preds)))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['accuracy'],
                                                    name='train_accuracy')
            self.visualizer.display_current_results(state.stage_epoch_step,
                                                    state.epoch_metrics['valid']['accuracy'], name='valid_accuracy')
class MetricsCallback_loc(Callback):

    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "bound_loss,classification_loss,acc_pre_rec_f1",
                 func = getMetrics):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.directory = directory
        self.model_name = model_name
        self.check_interval = check_interval
        self.func = func
        self.drawing=DataCreation(image_path_='/home/pooja/PycharmProjects/digitRecognizer/rough/localization/images')
        self.visualizer = Visualizer()

    # def on_batch_end(self,state: State):# #
    #     targ = state.input[self.input_key].detach().cpu().numpy()
    #     out = state.output[self.output_key]
    #
    #     clipwise_output = out[self.model_output_key].detach().cpu().numpy()
    #
    #     self.prediction.append(clipwise_output)
    #     self.target.append(targ)
    #
    #     y_pred = clipwise_output.argmax(axis=1)
    #     y_true = targ.argmax(axis=1)

    # score = f1_score(y_true, y_pred, average="macro")
    # state.batch_metrics[self.prefix] = score
    def draw_image(self, preds,msg= ""):
        i=0
        list_ = os.listdir(self.drawing.image_path)
        for img in list_:
            if img.find("pred") != -1:os.remove(self.drawing.image_path+"/"+img)
        list_ = os.listdir(self.drawing.image_path)
        list_.sort(key=lambda x: int(x.split("_")[0]))


        for img in list_[0:50]:
            self.drawing.draw_box(*preds[i,:].tolist(),data =None,color_intensity=(0,0,200),save_loc=self.drawing.image_path+"/"+img,msg =(msg[0][i], msg[1][i]))
            i+=1
            if i==10 :break

    def rub_pred(self):
        i=0
        # list_=os.listdir(self.drawing.image_path)
        # list_.sort(key=lambda x: int(x.split("_")[0]))
        # for img in list_[0:9]:
        #     self.drawing.rub_box(data =None,dim=2,save_loc=self.drawing.image_path+"/"+img)
        #     i+=1
    def on_epoch_end(self, state: Runner):
        """Event handler for epoch end.

        Args:
            runner ("IRunner"): IRunner instance.
        """
        #print(torch.sum(state.model.fc1.weight),state.model.fc1.weight[5][300])
        #print(torch.sum(state.model.conv_blocks[0].conv1.weight))
        if self.directory is not None: torch.save(state.model.state_dict(), str(self.directory) + '/' +
                                                  self.model_name + "_" + str(
            state.stage_epoch_step) + ".pth")

        if (state.stage_epoch_step + 1) % self.check_interval == 0:
            preds = state.batch['logits']
            pred_class= torch.argmax(state.batch['logits'][:,:10], dim=1)
            accuracy_metrics=getMetrics(state.batch['targets'][:, 0], pred_class)
            loss=self.func(state.batch['targets'], preds)
            print("{} is {}{}".format(self.prefix, loss,accuracy_metrics))
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['train']['loss'],
                                                    name='train_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, state.epoch_metrics['valid']['loss'],
                                                    name='valid_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, accuracy_metrics[0],
                                                    name='accuracy')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[0],
                                                    name='bounding_loss')
            self.visualizer.display_current_results(state.stage_epoch_step, loss[1],
                                                    name='classification_loss')

    def on_batch_end(self,state):
        # if state.global_batch_step == 1:
        #     self.rub_pred()
        torch.nn.utils.clip_grad_value_(state.model.parameters(), clip_value=1.0)
        if state.loader_batch_step==1 and (state.global_epoch_step-1)%5==0 and state.is_train_loader:
            preds = state.batch['logits']
            pred_class = torch.argmax(state.batch['logits'][:, :10], dim=1)
            max_prob=torch.max(state.batch['logits'][:, :10], dim=1)[0]
            #self.rub_pred()
            self.draw_image(preds[:, -4:], msg = (pred_class,max_prob))

            #print("max_gradient is "+ torch.max(state.model.state_dict().values()[0]))





