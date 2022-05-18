import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch
from torch_geometric.utils import to_networkx, from_networkx
import rewiring
import networkx as nx
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
    names = ["squirrel"]
    hyperparams = {
    "cornell": AttrDict({'dropout': 0.08056718510279537, 'num_layers': 1, 'dim': 128, 'learning_rate': 0.043259685998514125, 'weight_decay': 0.11441342982302352, 'max_iterations': 780}),
    "texas": AttrDict({"skip_connection": 0.1, "dropout": 0.5954, "num_layers": 1, "dim": 128, "learning_rate": 0.0278, "weight_decay": 0.0623, "max_iterations": 47, "temperature": 172, "C_plus": 2.25}),
    "wisconsin": AttrDict({"skip_connection": 0.1, "dropout": 0.6033, "num_layers": 1, "dim": 128, "learning_rate": 0.0295, "weight_decay": 0.1920, "max_iterations": 27, "temperature": 32, "C_plus": 0.5}),
    "chameleon": AttrDict({'dropout': 0.43446212772219317, 'num_layers': 1, 'dim': 64, 'learning_rate': 0.033660439322985264, 'weight_decay': 0.16331051837548435, 'max_iterations': 1842}),
    "squirrel": AttrDict({'dropout': 0.4983797463417786, 'num_layers': 1, 'dim': 128, 'learning_rate': 0.003585299305048884, 'weight_decay': 0.19801432057717588, 'max_iterations': 800}),
    "actor": AttrDict({'dropout': 0.8675000061339353, 'num_layers': 1, 'dim': 16, 'learning_rate': 0.04676186778179084, 'weight_decay': 0.0071627046187316135, 'max_iterations': 452}),
    "cora": AttrDict({'dropout': 0.3660401848946039, 'num_layers': 2, 'dim': 16, 'learning_rate': 0.031477568591226184, 'weight_decay': 0.20106306553056918, 'max_iterations': 69}),
    "citeseer": AttrDict({'dropout': 0.5978429814505746, 'num_layers': 2, 'dim': 32, 'learning_rate': 0.03843724741519129, 'weight_decay': 0.01564545403728318, 'max_iterations': 1818}),
    "pubmed": AttrDict({'dropout': 0.5254051009158754, 'num_layers': 2, 'dim': 128, 'learning_rate': 0.031847737302168076, 'weight_decay': 0.05360889354880405, 'max_iterations': 1710}),
    }
    stopping_criterion = STOP.VALIDATION
    num_trials = 100
    accuracies = []
    for name in names:
        accuracies = []
        print("TESTING: " + name)
        for trial in range(num_trials):
            print("Trial number: ", trial)
            dataset = task.get_dataset()
            dataset.generate_data(name)
            G = to_networkx(dataset.graph, to_undirected=True)
            #print("Starting spectral gap: ", rewiring.spectral_gap(G))
            for i in range(hyperparams[name].max_iterations):
                rewiring.rlef(G)
            dataset.graph.edge_index = from_networkx(G).edge_index
            args = main.get_fake_args(task=task, num_layers=hyperparams[name].num_layers, loader_workers=7,
                                      type=gnn_type, stop=stopping_criterion, dataset=dataset, last_layer_fully_adjacent=False)
            train_acc, validation_acc, test_acc, epoch = Experiment(args).run()
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("E(X^2): ", sum([x**2 for x in accuracies]))
        print("Accuracy for ")

        f = open("rlef-notes.txt", "a")
        f.write("\n" + "Average accuracy for " + name + ": " + np.average(accuracies))
        f.write("\n" + "E(X^2): " + str(sum([x**2 for x in accuracies])) + "\n")
        f.close()
