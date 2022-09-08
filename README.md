# Oversquashing in GNNs through the lens of information contraction and graph expansion

Code for the synthetic experiments in our paper on oversquashing in GNNs to appear in the [58th Annual Allerton Conference on Communication, Control, and Computing] (https://allerton.csl.illinois.edu/) (preprint: [arXiv:2208.03471](https://arxiv.org/abs/2208.03471)). 

## Requirements
To configure and activate the conda environment for this repository, run
```
conda env create -f environment.yml
conda activate oversquashing
```
## Spectral expansion plots
To test the spectral expansion of a graph under RLEF, G-RLEF, or SDRF, run the file plots.py. By default, it will produce a figure for a path of 3 cliques, each with 10 vertices.
## NeighborsMatch experiment
To run the NeighborsMatch experiment, run the file `run_neighborsmatch.py`. The following commands will run the experiment for the G-RLEF and SDRF rewirings.
```
python run_neighborsmatch.py --rewiring grlef
python run_neighborsmatch.py --rewiring sdrf
```
The full list of settings can be found in the file `hyperparams.py`.
