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
    names = ["cornell", "texas", "wisconsin", "chameleon", "squirrel", "actor", "cora", "citeseer", "pubmed"]
    stopping_criterion = STOP.VALIDATION
    num_layers=3
    num_trials=20
    num_flips=100
    accuracies = []    
    for name in names:
        accuracies = []
        print("TESTING: " + name)
        for trial in range(num_trials):
            dataset = task.get_dataset()
            dataset.generate_data(name)
            G = to_networkx(dataset.graph, to_undirected=True)
            #print("Starting spectral gap: ", rewiring.spectral_gap(G))
            for flip in range(num_flips):
                rewiring.augment_degree(G)
                #print("Ending spectral gap: ", rewiring.spectral_gap(G))
            dataset.graph.edge_index = from_networkx(G).edge_index
            args = main.get_fake_args(task=task, num_layers=num_layers, loader_workers=7,
                                      type=gnn_type, stop=stopping_criterion, dataset=dataset, last_layer_fully_adjacent=False)
            train_acc, validation_acc, test_acc, epoch = Experiment(args).run()
            accuracies.append(test_acc)
            torch.cuda.empty_cache()
        print("average acc: ", np.average(accuracies))
        print("plus/minus: ", 2 * np.std(accuracies)/(num_trials ** 0.5))
