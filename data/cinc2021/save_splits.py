import os
import numpy as np


import dataset_cinc2021
import cfg_cinc2021

if __name__ == "__main__":
    seed = 42
    cfg_cinc2021.set_seed(seed)

    save_path = "./seed{}".format(seed)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train = dataset_cinc2021.CINC2021Dataset(cfg_cinc2021.TrainCfg, training=True)
    train._load_all_data()

    train._signals.dump(os.path.join(save_path, "X_train.npy"), protocol=4)
    train._labels.dump(os.path.join(save_path, "y_train.npy"), protocol=4)

    test = dataset_cinc2021.CINC2021Dataset(cfg_cinc2021.TrainCfg, training=False)
    test._load_all_data()

    test._signals.dump(os.path.join(save_path, "X_val.npy"), protocol=4)
    test._labels.dump(os.path.join(save_path, "y_val.npy"), protocol=4)
