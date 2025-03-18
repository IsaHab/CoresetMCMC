# Master thesis: Habereder Isabella

This is the practical part of the master thesis Coreset Markov chain Monte Carlo. 
The code implements the proposed algorithm of [Chen and Campbell](https://proceedings.mlr.press/v238/chen24f.html).

Python version: Python 3.12

- For illustrating the effect of the coreset size on the behaviour of CoresetMCMC run experiment_coresetsize.py
    for the regression models and experiment_gaussian_location_coresetsize.py for the Gaussian location model.
- For illustrating the effect of the subsample size on the behaviour of CoresetMCMC run experiment_subsample.py
    for the regression models and experiment_gaussian_location_subsample.py for the Gaussian location model.
- For comparing CoresetMCMC with the uniform subsampling method run experiment_unif.py for the regression models 
    and experiment_gaussian_location_unif.py for the Gaussian location model.