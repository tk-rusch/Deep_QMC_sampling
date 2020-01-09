# Enhanced Generalization in Deep Learning using Quasi-Monte Carlo Sampling
This repository contains the implementation to reproduce the numerical experiments of the paper "Enhanced Generalization in Deep Learning using Quasi-Monte Carlo Sampling"


## Installation
Please make sure you have installed the requirements before executing the python scripts.

```bash
pip install numpy
pip install matplotlib
pip install pytorch
pip install pandas
pip install filelock 
```

## Neural network training: Ensemble and retraining

In the source folder run
`python train_networks.py 'experiment' 'training type' 'sampling method' 'input dimension' 'maximum learning iterations'`
