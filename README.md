# auto_oo: an autodifferentiable framework for molecular orbital-optimized variational quantum algorithms
Generate orbital optimized cost functions based on electronic structure Hamiltonians in an active space, based on automatic differentation. Powered by [PennyLane](https://github.com/PennyLaneAI) using either a [Pytorch](https://pytorch.org/) or [Jax](https://github.com/google/jax) backend. Molecular structure calculations are performed with [PySCF](https://pyscf.org/).
Supports gradients and hessians of hybrid cost functions based on orbital rotations and parameterized quantum circuits.
We use a parameterization of orbitals by rotating the orthogonal atomic orbitals (OAOs).
This code was developed as a tool to enable the calculation of Berry phases in molecular systems, see [arXiv:2304.06070](https://arxiv.org/abs/2304.06070).

## Install

1. create an anaconda or pip environment.
2. install PyTorch following the instructions in PyTorch's [Get Started](https://pytorch.org/get-started/locally/). (GPU support is not needed for small active spaces)
3. Clone the repository `git clone git@github.com:EmielKoridon/auto_oo.git` and move into the cloned directory `cd auto_oo`
4. Install auto_oo and its dependencies with `pip install .`

### Dependencies

Auto_oo requires the following dependencies:
- pytorch [this should be installed manually](https://pytorch.org/get-started/locally/)
- numpy
- pennylane (version >= 0.31.1, where the orbital-rotation bug is fixed. See [here](https://github.com/PennyLaneAI/pennylane/commit/5c87d88dfb36e8a173c97378e01ed6f40960d317).
- scipy
- openfermion
- pyscf
- jax

Furthermore in the tutorials, we use [cirq](https://github.com/quantumlib/cirq) to visualize states using Dirac notation.


## Tutorials and example

For an extensive tutorial of the code, see the examples section. For the moment, we provide two jupyer notebooks
1. `examples/Tutorial_auto_oo.ipynb` - in this introduction to the usage of the code, we implement a orbital-optimized PQC ansatz and optimize it using automatic differentation
2. `examples/Tutorial_Berry_phase.ipynb` - in this tutorial we show how to use the orbital-optimized PQC ansatz to detect a conical intersection by calculating the Berry phase with the algorithm described in [arXiv:2304.06070](https://arxiv.org/abs/2304.06070).


## Code structure

```
auto_oo
├── README.md
├── pyproject.toml
│
├── examples
│   ├── Tutorial_auto_oo.ipynb
│   └── Tutorial_Berry_phase.ipynb
│
├── src
│   └── auto_oo
│       ├── ansatze
│       │   ├── __init__.py
│       │   ├── kUpCCD.py
│       │   └── uccd.py
│       ├── __init__.py
│       ├── moldata_pyscf.py
│       ├── noisy_oo_pqc.py
│       ├── oo_energy.py
│       ├── oo_pqc.py
│       ├── pqc.py
│       └── utils
│           ├── active_space.py
│           ├── __init__.py
│           ├── miscellaneous.py
│           └── newton_raphson.py
└── test
    └── ...
```


