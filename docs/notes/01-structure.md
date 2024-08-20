# Package structure

Code should follow the `src/` layout, as [described here](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/), which "helps enforce that an editable installation is only able to import files that were meant to be importable."

This is recommended by both PyOpenSci and PyPA.

The structure should look like (copied from pyopensci):
```
myPackageRepoName
├── CHANGELOG.md               ┐
├── CODE_OF_CONDUCT.md         │
├── CONTRIBUTING.md            │
├── docs                       │ Package documentation
│   └── index.md
│   └── ...                    │
├── LICENSE                    │
├── README.md                  ┘
├── pyproject.toml             ] Package metadata and build configuration
├── src                        ┐
│   └── myPackage              │
│       ├── __init__.py        │ Package source code
│       ├── moduleA.py         │
│       └── moduleB.py         ┘
└── tests                      ┐
   └── ...                     ┘ Package tests
```

Note that you need to have `__init__.py` under `src/myPackage/` and every sub-directory.
