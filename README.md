# The Importance of Layer Ordering

This codebase accompanies the paper "The Effect of Layer Ordering in Hybrid Architectures". The experiments fall into roughly three categories:

1. Segmented Retrieval and Tensor Lookup. These experiments test the performance of hybrid architectures of various layer orderings for shallow hybrids.
2. Exact Copy. These experiments use an SSM variant with two gates rather than the single gate found in Mamba architectures. This follows from the construction.
3. Associtive Recall. These experiments include performance, but for deeper hybrids. Additionally, a variety of other experiments are conducted on this task to reveal the learned algorithms for various hybrids.

Each of these experiment categories are executed in slightly different ways.

### Segmented Retrieval and Tensor Lookup

For these two tasks, shell scripts can be found in `standard_archs/*.sh`. Each filename matches the task is executes. Memory is controlled as a parameter in each of these scripts. This suite also contains the experiment setup for running Selective Copying.

### Exact Copy

The folder `exact_copy` contains the relevant script, `main.sh`. This file shares roughly the same structure as for the other tasks, with memory controlled as a parameter in `main.sh`. This shell script expects two parameters: the hybrid architecture to run (e.g. `TST`) and the ratio of window width to state expansion factor (e.g. `7` for 7 to 1).

### Associative Recall

The folder `assoc_recall` contains python scripts that each execute different versions of the associative recall results.

## Other files

In the base directory exists a file titled `gather_results.py`. This python script takes the printed output of one (or many) of these shell scripts and aggregates the results into one `.json` file. The keys of this file contain the architecture, window width, state expansion factor, token dimension, and inference memory used by the model.
