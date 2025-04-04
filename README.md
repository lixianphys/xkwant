## About Kwant and why xKwant
Kwant is a powerful [open-source Python library](https://github.com/kwant-project) designed for numerical calculations on tight-binding models. Despite its convenient API, working with Kwant can still involve some repetitive and tedious tasks, especially for batch calculations and parameter tracking. This is where xKwant comes in.


## Pain Points of Using Kwant

- Tedious and Repetitive Coding: Despite Kwant's API, batch calculations often require writing similar blocks of code repeatedly, making the process time-consuming and error-prone.

- Parameter Tracking Challenges: When experimenting with multiple parameters, keeping track of which values were used in each run can be difficult, complicating reproducibility and analysis.

- Disorganized Scripts and Templates: Managing ad-hoc scripts separately from reusable templates and frequently used functions can be cumbersome. Proper organization is crucial for version control, especially when working with Git.

## How This Toolbox Helps

To address these challenges, this toolbox automates repetitive tasks and introduces a structured approach to handling complex models. 
It includes:

- Batch Calculation Templates: Simplifies running multiple simulations efficiently.

- Device Construction Tools: Streamlines the creation and modification of devices.

- Automated Input and Output Logging: Tracks parameters and results, ensuring better experiment reproducibility (xkwant.log and console).

By reducing manual effort, improving organization, and enhancing workflow efficiency, this toolbox makes building and analyzing tight-binding models in Kwant significantly more seamless.



## Core functionalities
```bash
scripts              
xkwant
├── batch.py         # batch calculation
├── benchmark.py     # benchmarking new templates
├── schemas.py       # data models
├── device.py        # templates to define a device in 2D space
├── log.py           # tracking parameters, inputs, and outputs
├── physics.py       # physics constants
├── templates.py     # templates to run calculations (reduce redundancy)
└── utils.py         # utilities
```

## Installation and Tests

### Install Kwant with conda 
```shell
conda env create -f env_kwant.yml
conda activate MyKwant
python -c"import kwant; print(f'Kwant version: {kwant.__version__}')"
```
### Install other dependencies
```shell
python -m pip install "numpy<2" scipy matplotlib tqdm sympy pytest
```
### Run tests locally
```shell
python -m pytest test.py
```