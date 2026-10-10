# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:18:02 2022

@author: nbagz

ShipD's isolated NumPy-1 input-admission boundary and in-memory mesh operation.
mesh(job) retains the dataset resolution (100 waterlines x 800 points); --lid closes a hull with ShipD's own deck lid.
The command consumes declared current-source rows on stdin and returns geometry on stdout.
The calling research producer owns Store persistence; no mesh/result files are written here.

    python STLGen.py --admit-only --input-format FORMAT --columns JSON --source PROVENANCE
    # JSON rows on stdin; admission declaration on stdout, no filesystem result.

Historical dataset vectors are admitted only with their recorded format/order and retired features disabled.
"""
import os

# One BLAS thread in the isolated vendor operation. Set before numpy loads.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path

import numpy as np

from HullParameterization import Hull_Parameterization as HP

HERE = Path(__file__).resolve().parent
MAX_ADMISSION_BYTES = 8 * 1024 * 1024  # bounded existing-command IPC, never a whole-data preflight


def mesh(job):
    """One declared hull as (index, triangles, error), without result delivery."""
    index, vector, input_format, columns, source, lid = job
    try:
        hull = HP.from_declared(vector, input_format=input_format, columns=columns,
                                source=source)
        triangles = hull.gen_stl(NUM_WL=100, PointsPerWL=800, bit_AddDeckLid=int(lid))
    except Exception as error:
        return index, None, f"{type(error).__name__}: {error}"
    return index, triangles, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1, help="requested worker count; this isolated in-memory command uses one worker")
    parser.add_argument("--lid", action="store_true", help="close each hull at its deck (the dataset's meshes are open)")
    parser.add_argument("--admit-only", action="store_true", help="admit JSON rows on stdin; never construct meshes")
    parser.add_argument("--input-format", help="the declared HP input format; not inferred from array width")
    parser.add_argument("--columns", help="JSON list of column names in their actual declared order")
    parser.add_argument("--source", help="input source provenance, separate from a scientific acceptance claim")
    args = parser.parse_args()
    try:
        payload = sys.stdin.buffer.read(MAX_ADMISSION_BYTES + 1)
        if len(payload) > MAX_ADMISSION_BYTES:
            raise ValueError("ShipD admission payload exceeds the bounded IPC byte limit")
        transport = json.loads(payload)
        if not isinstance(transport, dict) or set(transport) != {"rows", "shape"}:
            raise ValueError("ShipD admission IPC requires the rows/shape transport envelope")
        shape = transport["shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("ShipD admission IPC requires a nonnegative two-dimensional shape")
        rows = np.asarray(transport["rows"])
        if rows.size == 0:
            rows = rows.reshape(shape)
        elif rows.shape != tuple(shape):
            raise ValueError("ShipD admission IPC shape must match the original nested rows")
        columns = json.loads(args.columns) if args.columns is not None else None
        admitted = HP.admit(rows, input_format=args.input_format, columns=columns, source=args.source)
        declaration = {"status": "admitted-input-only", "input_format": args.input_format,
                       "columns": columns, "source": args.source, "rows": len(admitted),
                       "current_columns": HP.INPUT_COLUMNS}
        if args.admit_only:
            print(json.dumps(declaration))
            return
        if args.workers < 1:
            raise ValueError("ShipD worker count must be positive")
        # Sequential CPU work belongs to the existing isolated NumPy-1 operation,
        # not the GPU lattice path; stdout carries complete original-resolution meshes.
        payloads, errors = {}, []
        with redirect_stdout(sys.stderr):
            for index, vector in enumerate(rows):
                _, triangles, error = mesh((index, vector, args.input_format, columns, args.source, args.lid))
                errors.append(error)
                if triangles is not None:
                    payloads[f"triangles_{index}"] = triangles
        metadata = {"status": "in-memory-native-meshes-not-scientific-acceptance", "admission": declaration,
                    "errors": errors, "requested_workers": args.workers, "effective_workers": 1,
                    "waterlines": 100, "points_per_waterline": 800, "deck_lid": args.lid}
        encoded = io.BytesIO()
        np.savez(encoded, metadata=json.dumps(metadata, allow_nan=False), **payloads)
        sys.stdout.buffer.write(encoded.getvalue())
    except (ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
