# Enhancing accuracy of deep learning algorithms by training with low-discrepancy sequences
This repository contains the implementation to reproduce the numerical experiments of the paper "Enhancing accuracy of deep learning algorithms by training with low-discrepancy sequences"


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

     python train_networks.py 'experiment' 'training_type' 'sampling_method' 'input_dimension' 'max_learning_iterations'
     
**experiment**: name of the experiment type
- *BSPDE*, for the Black-Scholes PDE problem
- *airfoil*,  for the airfoil problem
- *projectile* for the projectile motion problem
- *sum_sines*, for sum of sines problem

**training_type**: name of the training procedure:
- *ensemble*, to perform an ensemble training
- *retrain*, to perform a retraining of the best performing network
```diff
! Note: only do a retraining once the corresponding MC ensemble training is finished. Everything else will lead to FileNotFound errors.
```

**sampling_method**: Sampling method for training and test set:
- *QMC*, for the Sobol-based Quasi-Monte Carlo sampling
- *MC*, for the standard Monte-Carlo sampling

**input_dimension**: integer to indicate the input space dimension
```diff
! Note: for every 'training_type' only the data for input dimensions specified in the experiments of the paper are available. Everything else will lead to FileNotFound errors.
```

**max_learning_iterations**: integer to indicate the maximum amount of learning iterations

The results are stored in `data/<experiment>/results`.
