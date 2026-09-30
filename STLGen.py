# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:18:02 2022

@author: nbagz

Meshes ShipD hulls to STL in parallel: a pool of processes, each building a hull from its design vector and writing
Hull_mesh_<index>.stl at the dataset's resolution (100 waterlines x 800 points).

    python STLGen.py OUTPUT [--vectors InputVectors_30k.npy] [--count N] [--workers W]

A Constrained Randomized Set is meshed with --vectors Constrained_Randomized_Set_<j>/Input_Vectors.csv and its stl/ folder
as OUTPUT.
"""
import os

# one BLAS thread per process: the pool is the parallelism. Set before numpy loads, here and in every worker.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from HullParameterization import Hull_Parameterization as HP

HERE = Path(__file__).resolve().parent


def mesh(job):
    """Meshes one hull; a hull whose construction fails (a singular solve, say) comes back as its error."""
    index, vector, output = job
    try:
        hull = HP(vector)
        hull.save_stl(hull.gen_stl(NUM_WL=100, PointsPerWL=800), namepath=str(output / f"Hull_mesh_{index}"))
    except Exception as error:
        return index, f"{type(error).__name__}: {error}"
    return index, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="the folder the STL files go to")
    parser.add_argument("--vectors", type=Path, default=HERE / "InputVectors_30k.npy",
                        help="the design vectors: a .npy array, or a .csv with one vector per row")
    parser.add_argument("--count", type=int, help="mesh only the first COUNT hulls")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes meshing at once")
    args = parser.parse_args()
    vectors = np.load(args.vectors) if args.vectors.suffix == ".npy" else np.loadtxt(args.vectors, delimiter=",")
    vectors = vectors[:args.count].astype(np.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(mesh, [(i, vector, args.output) for i, vector in enumerate(vectors)]), total=len(vectors)))
    failed = [(index, error) for index, error in results if error]
    print(f"{len(vectors) - len(failed)} of {len(vectors)} hulls in {time.perf_counter() - start:.1f} s with {args.workers} workers")
    for index, error in failed:
        print(f"    hull {index}: {error}")


if __name__ == "__main__":
    main()
