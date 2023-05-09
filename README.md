# auto_oo
Generate orbital optimized cost functions and Hamiltonians based on automatic differentation by [Pytorch](https://pytorch.org/).
Supports gradients and hessians of hybrid cost functions based on orbital rotations and parameterized quantum circuits.
We use a parameterization of orbitals by rotating the orthogonal atomic orbitals (OAOs).

## Install
First, create an anaconda or pip environment and install PyTorch, for example locally in a cpu-only installation:\
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`\
Optionally use GPU if your system supports it.\
Then, install this package by cloning the repo, then use `pip install -e .` in the cloned directory.

## Code structure
```
auto_oo
├── examples
│   ├── formaldimine.png
│   ├── three_loops_FCI.png
│   └── Tutorial.ipynb
├── pyproject.toml
├── README.md
└── src
    └── auto_oo
        ├── ansatze
        │   ├── __init__.py
        │   ├── kUpCCD.py
        │   └── uccd.py
        ├── __init__.py
        ├── moldata_pyscf.py
        ├── noisy_oo_pqc.py
        ├── oo_energy.py
        ├── oo_pqc.py
        ├── pqc.py
        └── utils
            ├── active_space.py
            ├── __init__.py
            └── newton_raphson.py

```

## To-do:
- Newton-Raphson steps with trust-region method
- More efficient implementation of RDMs. Measure e_pqrs FermionOperators directly on circuit?
- More efficient implementation of OO Hessian, not necessary to compute full space RDMs.
