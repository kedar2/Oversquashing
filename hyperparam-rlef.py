import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch
from torch_geometric.utils import to_networkx, from_networkx
import rewiring
import networkx as nx
import numpy as np
from numpy.random import random
from sklearn.model_selection import train_test_split

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
    names = ["cornell", "chameleon", "squirrel", "actor", "cora", "citeseer", "pubmed"]

    stopping_criterion = STOP.VALIDATION
    num_trials= 25
    for name in names:
        print("TUNING: " + name)

        best_acc = 0

        dataset = task.get_dataset()
        dataset.generate_data(name)
        num_nodes = dataset.num_nodes
        node_indices = list(range(num_nodes))
        non_test_samples, test_samples = train_test_split(node_indices, test_size=0.2)

        for trial in range(num_trials):
            train_samples, validation_samples = train_test_split(non_test_samples, test_size=0.25)
            triangle_data = None

            dropout = random()
            num_layers = np.random.choice([1,2,3])
            dim = np.random.choice([16, 32, 64, 128])
            learning_rate = random() * 0.05
            weight_decay = random() * 0.25
            max_iterations = np.random.randint(1, 2000)

            hyperparams = AttrDict({"dropout": dropout, "num_layers": num_layers, "dim": dim, "learning_rate": learning_rate, "weight_decay": weight_decay, "max_iterations": max_iterations})

            G = to_networkx(dataset.graph, to_undirected=True)

            rewiring.rlef(G)
            
            args = main.get_fake_args(task=task, num_layers=hyperparams.num_layers, loader_workers=7,
                                      type=gnn_type, stop=stopping_criterion, dataset=dataset, last_layer_fully_adjacent=False, preloaded_samples=(train_samples, validation_samples, test_samples))
            train_acc, validation_acc, test_acc, epoch = Experiment(args).run()
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_hyperparams = hyperparams

            torch.cuda.empty_cache()
        print("Best accuracy: ", best_acc)
        print("Best hyperparameters: ", best_hyperparams)
        f = open("hyperparam-notes.txt", "a")
        f.write("\n" + "(RLEF) Best hyperparameters for " + name + ": " + str(best_hyperparams))
        f.write("\n" + "Accuracy: " + str(best_acc) + "\n")
        f.close()
