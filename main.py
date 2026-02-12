import numpy as np
from HullParameterization import Hull_Parameterization as HP

Vectors = np.load('./scripts/ShipD/InputVectors_30k.npy')

def globalOptions(shipdIndex: int = 0, shipdDwgSclFct: float = 20.0, shipdWLNum: int = 100, shipdWLPts: int = 800,
                  shipdXsom: bool = True, shipdDeck: bool = True, shipdVector: bool = False, **kwargs):
    global OPTIONS
    OPTIONS = {
        "vector_index":   shipdIndex,
        "scale_factor":   shipdDwgSclFct,
        "NUM_WL":         shipdWLNum,
        "PointsPerWL":    shipdWLPts,
        "bit_AddTransom": shipdXsom,
        "bit_AddDeckLid": shipdDeck,
        "vector":         shipdVector
    }

def main(Lb: float = 0.48, Ls: float = 0.45, Bd: float = 0.21, Dd: float = 0.15, Bs: float = 0.5, WL: float = 0.5,
              Bc: float = 0.28, Beta: float = 22.5, Rc: float = 0.5, Rk: float = 0.0, BOWA: float = 0.0,
              BOWB: float = 0.0, BK: float = 0.5, Kappa_BOW: float = 0.5, DELTA_BOWA: float = 0.0,
              DELTA_BOWB: float = 0.0, DRIFTA: float = 0.0, DRIFTB: float = 0.0, DRIFTC: float = 30.0,
              bit_EP_S: bool = True, bit_EP_T: bool = True, TRANSA: float = 1.0, SK: float = 0.5,
              Kappa_STERN: float = 0.5, DELTA_STERNA: float = 0.0, DELTA_STERNB: float = 0.0, Beta_trans: float = 30.0,
              Bc_trans: float = 0.25, Rc_trans: float = 0.25, Rk_trans: float = 0.0, bit_BB: bool = True,
              bit_SB: bool = True, Lbb: float = 0.1, Hbb: float = 0.5, Bbb: float = 0.5, Lbbm: float = 0.0,
              Rbh: float = 0.19, Kappa_SB: float = 0.5, Lsb: float = 0.1, HsbOA: float = 0.5, Hsb: float = 0.5,
              Bsb: float = 0.5, Lsbm: float = 0.0, Rsb: float = 0.19, **kwargs):
    if OPTIONS["vector"]:
        vector = Vectors[OPTIONS["vector_index"]]
    else:
        vector = np.array([
            10, Lb, Ls, Bd, Dd, Bs, WL, Bc, Beta, Rc, Rk, BOWA, BOWB, BK, Kappa_BOW, DELTA_BOWA, DELTA_BOWB,
            DRIFTA, DRIFTB, DRIFTC, int(bit_EP_S), int(bit_EP_T), TRANSA, SK, Kappa_STERN, DELTA_STERNA,
            DELTA_STERNB, Beta_trans, Bc_trans, Rc_trans, Rk_trans, int(bit_BB), int(bit_SB), Lbb, Hbb,
            Bbb, Lbbm, Rbh, Kappa_SB, Lsb, HsbOA, Hsb, Bsb, Lsbm, Rsb
        ])


    Hull = HP(vector)
    strpath = './Sample_Hull_Mesh'
    hullvectors = Hull.gen_stl(NUM_WL=OPTIONS["NUM_WL"], PointsPerWL=OPTIONS["PointsPerWL"], bit_AddTransom=OPTIONS["bit_AddTransom"], bit_AddDeckLid=OPTIONS["bit_AddDeckLid"], namepath=strpath)
    hullvectors *= OPTIONS["scale_factor"]
    return hullvectors
