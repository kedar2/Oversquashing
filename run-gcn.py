import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch
import numpy as np

override_params = {
    2: {'batch_size': 64, 'eval_every': 1000},
    3: {'batch_size': 64},
    4: {'batch_size': 1024},
    5: {'batch_size': 1024},
    6: {'batch_size': 1024},
    7: {'batch_size': 2048},
    8: {'batch_size': 1024, 'accum_grad': 2},  # effective batch size of 2048, with less GPU memory
}


class Results:
    def __init__(self, train_acc, test_acc, epoch):
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.epoch = epoch


if __name__ == '__main__':

    task = Task.DEFAULT
    gnn_type = GNN_TYPE.GCN
    names = ["cornell", "texas", "wisconsin", "chameleon", "squirrel", "actor", "cora", "citeseer", "pubmed"]
    hyperparams = {
    "cornell": AttrDict({"dropout": 0.3060, "num_layers": 1, "dim": 128, "learning_rate": 0.0082, "weight_decay": 0.1570}),
    "texas": AttrDict({"dropout": 0.2346, "num_layers": 1, "dim": 128, "learning_rate": 0.0072, "weight_decay": 0.0037}),
    "wisconsin": AttrDict({"dropout": 0.2869, "num_layers": 1, "dim": 64, "learning_rate": 0.0281, "weight_decay": 0.1570}),
    "chameleon": AttrDict({"dropout": 0.7304, "num_layers": 3, "dim": 128, "learning_rate": 0.0248, "weight_decay": 0.0936}),
    "squirrel": AttrDict({"dropout": 0.5974, "num_layers": 2, "dim": 64, "learning_rate": 0.0136, "weight_decay": 0.1346}),
    "actor": AttrDict({"dropout": 0.7605, "num_layers": 1, "dim": 64, "learning_rate": 0.0290, "weight_decay": 0.0619}),
    "cora": AttrDict({"dropout": 0.4144, "num_layers": 1, "dim": 64, "learning_rate": 0.0097, "weight_decay": 0.0639}),
    "citeseer": AttrDict({"dropout": 0.7477, "num_layers": 1, "dim": 128, "learning_rate": 0.0251, "weight_decay": 0.4577}),
    "pubmed": AttrDict({"dropout": 0.4013, "num_layers": 1, "dim": 64, "learning_rate": 0.0095, "weight_decay": 0.0448})
    }
    stopping_criterion = STOP.VALIDATION
    num_layers=3
    num_trials=20
    for name in names:
        accuracies = []
        print("TESTING: " + name)
        for trial in range(num_trials):
            dataset = task.get_dataset()
            dataset.generate_data(name)
            args = main.get_fake_args(task=task, num_layers=num_layers, loader_workers=7,
                                      type=gnn_type, stop=stopping_criterion, dataset=dataset, last_layer_fully_adjacent=False)
            train_acc, validation_acc, test_acc, epoch = Experiment(args).run()
            args += hyperparams[name]
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("plus/minus: ", 2 * np.std(accuracies)/(num_trials ** 0.5))
    
