from dataLoaders import DigitData
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from config import *
from funcs import get_dict_from_class
from models import FeatureExtractor,FCLayered
from losses import BCELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold
import torch
def train(Model1,DataLoad1):
    randSeed=23
    data_load = DigitData(**get_dict_from_class(DataLoad1))
    criterion = BCELoss()
    model = FeatureExtractor(**get_dict_from_class(Model1))
    # model = FCLayered(**get_dict_from_class(Model1))
    if False:
        checkpoint = torch.load(str(saveDirectory) + '/featureExtr_4_100.pth')
        model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=randSeed)
    train = data_load.data
    train["fold"] = -1

    # train.set_index('index',inplace=True)
    for fold_id, (train_index, val_index) in enumerate(skf.split(train, train["fold"])):
        train.iloc[val_index, -1] = fold_id

    # # check the proportion
    fold_proportion = pd.pivot_table(train, columns="fold", values="label", aggfunc=len)

    use_fold = 0

    train_file = train.query("fold != @use_fold")
    val_file = train.query("fold == @use_fold")

    print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file), len(val_file)))

    loaders = {
        "train": DataLoader(DigitData(data_frame=train_file, **get_dict_from_class(DataLoad1)),
                            batch_size=512,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False),
        "valid": DataLoader(DigitData(data_frame=val_file, **get_dict_from_class(DataLoad1)),
                            batch_size=512,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    }

    callbacks = [
        dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=10, topk_args=[1]),

        MetricsCallback(input_key="targets", output_key="logits",
                        directory=saveDirectory, model_name='featureExtr_4'),
        # CheckpointCallback(save_n_best=0)
    ]
    runner = SupervisedRunner(

        output_key="logits",
        input_key="image_pixels",
        target_key="targets")
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=epoch,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
    )

    # main_metric = "epoch_f1",
    # minimize_metric = False
    c = 0
if __name__ == "__main__":
    train(Model1,DataLoad1)