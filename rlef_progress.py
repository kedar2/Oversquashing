import main
from common import Task, STOP, GNN_TYPE
from attrdict import AttrDict
from experiment import Experiment
import torch
from torch_geometric.utils import to_networkx, from_networkx
import rewiring
import networkx as nx
import numpy as np
import pickle

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
    names = ["cornell", "texas", "wisconsin"]
    hyperparams = {
    "cornell": AttrDict({"dropout": 0.2411, "num_layers": 1, "dim": 128, "learning_rate": 0.0172, "weight_decay": 0.0125, "max_iterations": 25, "temperature": 130, "C_plus": 0.25}),
    "texas": AttrDict({"dropout": 0.5954, "num_layers": 1, "dim": 128, "learning_rate": 0.0278, "weight_decay": 0.0623, "max_iterations": 47, "temperature": 172, "C_plus": 2.25}),
    "wisconsin": AttrDict({"dropout": 0.6033, "num_layers": 1, "dim": 128, "learning_rate": 0.0295, "weight_decay": 0.1920, "max_iterations": 27, "temperature": 32, "C_plus": 0.5}),
    "chameleon": AttrDict({"dropout": 0.7265, "num_layers": 1, "dim": 128, "learning_rate": 0.0180, "weight_decay": 0.2101, "max_iterations": 832, "temperature": 77, "C_plus": 3.35}),
    "squirrel": AttrDict({"dropout": 0.7401, "num_layers": 2, "dim": 16, "learning_rate": 0.0189, "weight_decay": 0.2255, "max_iterations": 6157, "temperature": 178, "C_plus": 0.5}),
    "actor": AttrDict({"dropout": 0.6866, "num_layers": 1, "dim": 128, "learning_rate": 0.0095, "weight_decay": 0.0727, "max_iterations": 1010, "temperature": 69, "C_plus": 1.22}),
    "cora": AttrDict({"dropout": 0.3396, "num_layers": 1, "dim": 128, "learning_rate": 0.0244, "weight_decay": 0.1076, "max_iterations": 100, "temperature": 163, "C_plus": 0.95}),
    "citeseer": AttrDict({"dropout": 0.4103, "num_layers": 1, "dim": 64, "learning_rate": 0.0199, "weight_decay": 0.4551, "max_iterations": 84, "temperature": 180, "C_plus": 0.22}),
    "pubmed": AttrDict({"dropout": 0.3749, "num_layers": 3, "dim": 128, "learning_rate": 0.0112, "weight_decay": 0.0138, "max_iterations": 166, "temperature": 115, "C_plus": 14.43}),
    }
    stopping_criterion = STOP.VALIDATION
    num_trials=100
    num_iterations=2000
    experiment_results = {}
    for name in names:
        accuracies = np.zeros(num_iterations)
        spectral_gaps = np.zeros(num_iterations)
        num_triangles = np.zeros(num_iterations)
        dataset = task.get_dataset()
        dataset.generate_data(name)
        curvatures = None
        G = to_networkx(dataset.graph, to_undirected=True)
        for trial in range(num_trials):
            spectral_gaps[trial] += rewiring.spectral_gap(G)
            num_triangles[trial] += rewiring.number_of_triangles(G)
            rewiring.rlef(G)
        accuracies /= num_trials
        spectral_gaps /= num_trials
        num_triangles /= num_trials
        experiment_results[name] = {"accuracies": accuracies, "spectral_gaps": spectral_gaps, "num_triangles": num_triangles}
    pickle.dump(experiment_results, open("experiments.p" ,"wb"))