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
├── src
│   └── auto_oo
│       ├── ansatz
│       │   ├── kUpCCD.py
│       │   ├── pqc.py
│       │   └── uccd.py
│       ├── moldata_pyscf
│       │   └── moldata_pyscf.py
│       ├── newton_raphson
│       │   └── newton_raphson.py
│       ├── oo_energy
│       │   ├── integrals.py
│       │   └── oo_energy.py
│       └── oo_pqc
│           ├── noisy_oo_pqc.py
│           └── oo_pqc.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## To-do:
- Newton-Raphson steps with trust-region method
- More efficient implementation of RDMs. Measure e_pqrs FermionOperators directly on circuit?
- More efficient implementation of OO Hessian, not necessary to compute full space RDMs.
