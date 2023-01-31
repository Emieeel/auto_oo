# auto_oo
Orbital optimization based on automatic differentation by [Pytorch](https://pytorch.org/).
We use a parameterization of orbitals by rotating the orthogonal atomic orbitals (OAOs).

## Install
First, create an anaconda or pip environment and install PyTorch, for example locally in a cpu-only installation:\
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`\
Optionally use GPU if your system supports it.\
Then, install this package by cloning the repo, then use `pip install -e .` in the cloned directory.

## To-do at some point:
- Fast analytical orbital gradients and Hessians using generalized fock-matrices.
- SCF procedure with Newton-Raphson OO with RDMs

