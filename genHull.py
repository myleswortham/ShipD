import numpy as np
from HullParameterization import Hull_Parameterization as HP

Vectors = np.loadtxt('./scripts/ShipD/Input_Vectors_SampleHulls.csv', delimiter=",", dtype=np.float64)


def generate_mesh(vector_index = 0, scale_factor = 20.0, vector = None):
    if vector is None:
        vector = Vectors[vector_index]
    Hull = HP(vector)
    constraints = Hull.input_Constraints()
    cons = constraints > 0
    strpath =  './Sample_Hull_Mesh'
    mesh = Hull.gen_stl(NUM_WL=100, PointsPerWL=800, bit_AddTransom = 1, bit_AddDeckLid = 1, namepath = strpath)
    mesh.vectors *= scale_factor  # Scale the numpy-stl mesh
    return mesh
