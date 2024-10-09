## About Kwant and why xKwant
Kwant is a powerful open-source Python library designed for numerical calculations on tight-binding models. Its flexibility allows users and developers to build custom tools tailored for specific needs, which can also benefit a broader community of users. Despite its convenient API, working with Kwant can still involve some repetitive and tedious tasks, especially for batch calculations and parameter tracking. This is where xKwant comes in. 
## Pain points of using Kwant
While Kwant provides a convenient API, users often encounter the following challenges:
- Tedious and Repetitive Code: Despite Kwant's API, performing batch calculations often requires writing similar blocks of code repeatedly, which can be time-consuming and error-prone.
- Parameter Tracking: When experimenting with multiple parameters, it becomes challenging to keep track of which parameters were used in each run, making it difficult to reproduce or analyze results effectively.
- Separation of Ad-Hoc Scripts and Templates: Managing ad-hoc scripts separately from reusable templates and frequently needed functions is another pain point. This separation is crucial for better organization and version control, especially when working with Git.
## How to use this tool box
The tools I’ve developed aim to address these pain points by automating repetitive tasks and providing structured ways to handle complex models. This toolbox includes:

- Batch Calculations: Templates to streamline batch processing of simulations.
- Device Construction: Templates to build and modify devices.
- Input and Output Tracking: Tools for automatically logging parameters and results of successful runs, making it easier to track experiments.

These tools significantly reduce the effort required to build, analyze, and manage tight-binding models in Kwant, providing a more efficient workflow for complex simulations.

#### Core functionalities
```bash
xkwant
├── batch.py         # batch calculation
├── benchmark.py     # benchmarking new templates
├── config.py        # configuration settings
├── device.py        # templates to define a device in 2D space
├── log.py           # tracking parameters, inputs, and outputs
├── physics.py       # physics constants
├── plot.py          # plotting tools
├── templates.py     # templates to run calculations (reduce redundancy)
└── utils.py         # utilities
```
#### Scripts
store your script here.

#### plots
store your plots here (no git)

#### data
store your output files (.pkl) here (no git)
