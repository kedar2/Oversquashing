import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch
import numpy as np
import rewiring

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
    "cornell": AttrDict({"dropout": 0.6294, "num_layers": 1, "dim": 64, "learning_rate": 0.0134, "weight_decay": 0.0258, "alpha": 0.1795, "k": 64, "eps": None}),
    "texas": AttrDict({"dropout": 0.2382, "num_layers": 2, "dim": 128, "learning_rate": 0.0063, "weight_decay": 0.0153, "alpha": 0.0206, "k": 32, "eps": None}),
    "wisconsin": AttrDict({"dropout": 0.2941, "num_layers": 1, "dim": 128, "learning_rate": 0.0226, "weight_decay": 0.0226, "alpha": 0.1246, "k": None, "eps": 0.0001}),
    "chameleon": AttrDict({"dropout": 0.4191, "num_layers": 1, "dim": 128, "learning_rate": 0.0001, "weight_decay": 0.0001, "alpha": 0.0244, "k": 64, "eps": None}),
    "squirrel": AttrDict({"dropout": 0.7094, "num_layers": 1, "dim": 64, "learning_rate": 0.0192, "weight_decay": 0.0192, "alpha": 0.1610, "k": 64, "eps": 0.0016}),
    "actor": AttrDict({"dropout": 0.4012, "num_layers": 1, "dim": 64, "learning_rate": 0.0141, "weight_decay": 0.0141, "alpha": 0.0706, "k": None, "eps": None}),
    "cora": AttrDict({"dropout": 0.3315, "num_layers": 1, "dim": 64, "learning_rate": 0.0572, "weight_decay": 0.0572, "alpha": 0.0773, "k": 128, "eps": 0.0008}),
    "citeseer": AttrDict({"dropout": 0.5561, "num_layers": 1, "dim": 64, "learning_rate": 0.5013, "weight_decay": 0.5013, "alpha": 0.1076, "k": None, "eps": None}),
    "pubmed": AttrDict({"dropout": 0.4915, "num_layers": 2, "dim": 128, "learning_rate": 0.0597, "weight_decay": 0.0597, "alpha": 0.1155, "k": 128, "eps": None})
    }
    stopping_criterion = STOP.VALIDATION
    num_trials=20
    for name in names:
        accuracies = []
        print("TESTING: " + name)
        dataset = task.get_dataset()
        dataset.generate_data(name)
        args = main.get_fake_args(task=task, num_layers=hyperparams[name].num_layers, loader_workers=7,
            type=gnn_type, stop=stopping_criterion, dataset=dataset, last_layer_fully_adjacent=False)
        alpha = hyperparams[name].alpha
        k = hyperparams[name].k
        eps = hyperparams[name].eps
        dataset.data = rewiring.digl(dataset.graph, alpha=alpha, k=k, eps=eps)
        for trial in range(num_trials):
            train_acc, validation_acc, test_acc, epoch = Experiment(args).run()
            args += hyperparams[name]
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("plus/minus: ", 2 * np.std(accuracies)/(num_trials ** 0.5))
    
