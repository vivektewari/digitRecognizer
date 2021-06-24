from catalyst.dl import Callback, Runner, CallbackOrder
import torch
from funcs import getMetrics
from utils.visualizer import Visualizer


class MetricsCallback(Callback):
    def __init__(self,
                 directory=None,
                 model_name='',
                 check_interval=1,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "acc_pre_rec_f1"):
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
