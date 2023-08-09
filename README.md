# auto_oo
Generate orbital optimized cost functions based on electronic structure Hamiltonians in an active space, based on automatic differentation. Powered by [PennyLane](https://github.com/PennyLaneAI) using either a [Pytorch](https://pytorch.org/) or [Jax](https://github.com/google/jax) backend.
Supports gradients and hessians of hybrid cost functions based on orbital rotations and parameterized quantum circuits.
We use a parameterization of orbitals by rotating the orthogonal atomic orbitals (OAOs).
For an extensive tutorial of the code, see the examples section. This code was made to enable the calculation of Berry phases in molecular systems, see [our paper](https://arxiv.org/abs/2304.06070). In the examples folder you will also find a tutorial detailing how to compute Berry phases with `auto_oo`.

## Install
First, create an anaconda or pip environment and install PyTorch, for example locally in a cpu-only installation.\
Installation instructions can be found in the [Get Started](https://pytorch.org/get-started/locally/) section of the PyTorch website.
Optionally use GPU if your system supports it, but is not recommended for small active spaces.\
For now, an additional dependency is jax, of which the cpu version can be easily installed via pip. See the [Installation instruction](https://github.com/google/jax#installation).
The code is build such that you can use them interchangably.
Finally, install this package by cloning the repo, then use `pip install -e .` in the cloned directory.
### Dependencies
We have the following dependencies:
- numpy
- pennylane (version >= 0.31.1, where the orbital-rotation bug is fixed. See [here](https://github.com/PennyLaneAI/pennylane/commit/5c87d88dfb36e8a173c97378e01ed6f40960d317).
- scipy
- openfermion
- pyscf
- pytorch
- jax

In the tutorials, we use [cirq](https://github.com/quantumlib/cirq) to visualize states using Dirac notation.

## Code structure
```
auto_oo
├── examples
│   ├── formaldimine.png
│   ├── three_loops_FCI.png
│   ├── Tutorial_auto_oo.ipynb
│   └── Tutorial_Berry_phase.ipynb
├── pyproject.toml
├── README.md
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
    ├── test_moldata_pyscf.py
    ├── test_noisy_oo_pqc.py
    ├── test_oo_energy.py
    ├── test_oo_pqc.py
    ├── test_pqc.py
    └── utils
        ├── test_active_space.py
        ├── test_miscellaneous.py
        └── test_newton_raphson.py
```


