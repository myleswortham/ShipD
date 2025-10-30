import numpy as np
from HullParameterization import Hull_Parameterization as HP

Vectors = np.loadtxt('./Input_Vectors_SampleHulls.csv', delimiter=",", dtype=np.float64)

#Create one hull: 
Hull = HP(Vectors[0])

#Check Constraints:
constraints = Hull.input_Constraints()
cons = constraints > 0

#make the .stl file of the hull:
strpath =  './Sample_Hull_Mesh' 

mesh = Hull.gen_stl(NUM_WL=100, PointsPerWL=800, bit_AddTransom = 1, bit_AddDeckLid = 1, namepath = strpath)