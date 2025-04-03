"""
This benchmark is to access the performance and scalability of kwant model.
Key parameters:
Model_size N: the number of grid units in length direction

run python benchmark.py template_name (include all templates in xkwant/templates.py) model_sizes (a Python list)
"""

import time
import psutil

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import click
import ast

from xkwant.batch import *
from xkwant.templates import *
from xkwant.physics import *
from xkwant.utils import *
import xkwant.templates as mytemplates
from xkwant.schemas import GeomParams, HamParams


def benchmark_model(template, model_size=10):
    process = psutil.Process(os.getpid())
    start_time = time.time()
    memory_before = process.memory_info().rss

    energy_range = np.arange(0.1, 0.5, 0.01)
    N1 = model_size
    L = LATTICE_CONST_HGTE * N1
    geop = GeomParams(
        a=L / N1,
        lx_leg=int(N1),
        ly_leg=int(N1 / 6),
        lx_neck=int(N1 / 6),
        ly_neck=int(N1 / 6),
    )

    hamp_sys = HamParams(hop=0, mass=0.1, soc=0.3, wilson=0.05, inv=0, hyb=0)
    hamp_lead = HamParams(hop=0, mass=0.1, soc=0.3, wilson=0.05, inv=0, hyb=0)

    syst = template(geop, hamp_sys, hamp_lead)  # This system won't be changed anymore
    get_idos(syst, energy_range)
    # energy = np.mean(energy_range)
    # vvector_4t(syst, energy, [0, 0, Iin, -Iin])
    # rho_j_energy_site(syst, energy)

    end_time = time.time()

    memory_after = process.memory_info().rss

    execution_time = end_time - start_time
    memory_usage = memory_after - memory_before
    num_sites = syst.area / geop.a ** 2

    return num_sites, execution_time, memory_usage


@click.command()
@click.argument("template_name", required=False, default="mkhbar_4t")
@click.argument("model_sizes", required=False, default="[10,20,30,40,50]")
def benchmark(template_name,model_sizes):
    model_sizes = ast.literal_eval(model_sizes)
    try:
        template = getattr(mytemplates, template_name)
        click.echo(f"Benchmarking template: {template_name}")

    except AttributeError:
        click.echo(f"Template {template_name} not found in the module xkwant.templates", err=True)
        raise SystemExit(1)
    results = []
    # cold start to run the first time, to avoid the overhead of the first run
    _ = benchmark_model(template,10)
    for model_size in model_sizes:
        print(f"running model_size={model_size}")
        num_sites, execution_time, memory_usage = benchmark_model(template,model_size)
        results.append(
            {
                "model_size": model_size,
                "num_sites": num_sites,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
            }
        )

    df = pd.DataFrame(results)
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.scatter(
        df["model_size"].tolist()[1:],
        df["execution_time"].tolist()[1:],
        label="Execution Time ",
    )
    ax2.scatter(
        df["num_sites"].tolist()[1:],
        df["execution_time"].tolist()[1:],
        label="Memory Usage",
    )
    ax1.set_xlabel("Model Size in 1D")
    ax2.set_xlabel("Number of Sites (2D)")
    ax1.set_ylabel("Execution Time [Second]")
    ax2.set_ylabel("Execution Time [Second]")
    plt.show()


if __name__ == "__main__":
    benchmark()
