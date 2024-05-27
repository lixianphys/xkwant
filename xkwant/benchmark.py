# This benchmark is to access the performance and scalability of kwant model.
# Key parameters:
# Model_size N: the number of grid units in length direction

import time
import psutil

import kwant
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .batch import *
from .templates import *
from .physics import *
from .utils import *


def benchmark_model(model_size):
    process = psutil.Process(os.getpid())
    start_time = time.time()
    memory_before = process.memory_info().rss

    energy_range = np.arange(0, 0.5, 0.01)
    Iin = 10e-9  # A
    N1 = model_size
    L = 90 / 36 * N1
    geop = dict(
        lx_leg=int(N1), ly_leg=int(N1 / 6), lx_neck=int(N1 / 6), ly_neck=int(N1 / 6)
    )

    hamp_sys = dict(ts=0, ws=0.1, vs=0.3, ms=0.05, Wdis=0, a=L / N1)
    hamp_lead = dict(tl=0, wl=0.1, vl=0.3, ml=0.05)

    syst = mkhbar_4t(geop, hamp_sys, hamp_lead)  # This system won't be changed anymore
    density_to_energy(*get_idos(syst, energy_range), 0.01)
    energy = np.mean(energy_range)
    vvector_4t(syst, energy, [0, 0, Iin, -Iin])
    rho_j_energy_site(syst, energy)

    end_time = time.time()

    memory_after = process.memory_info().rss

    execution_time = end_time - start_time
    memory_usage = memory_after - memory_before
    num_sites = syst.area

    return num_sites, execution_time, memory_usage


def run_benchmark(model_sizes):
    results = []
    for model_size in model_sizes:
        num_sites, execution_time, memory_usage = benchmark_model(model_size)
        results.append(
            {
                "model_size": model_size,
                "num_sites": num_sites,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
            }
        )
    return results


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    model_sizes = np.arange(10, 80, 10, dtype=int)
    results = run_benchmark(model_sizes)
    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(
        df["model_size"].tolist(),
        df["execution_time"].tolist(),
        label=f"Execution Time",
    )
    ax2.plot(
        df["num_sites"].tolist(), df["execution_time"].tolist(), label=f"Memory Usage"
    )

    ax1.set_xlabel("Model Size")
    ax2.set_xlabel("Number of Sites")
    ax1.set_ylabel("Execution Time")
    ax2.set_ylabel("Execution Time")

    # plt.legend()
    plt.title("Benchmark Results")
    plt.show()
