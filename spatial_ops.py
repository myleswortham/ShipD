"""ShipD's ops for SpatialAI: a current31 hull (a mesh root), meshed when the app shows it."""
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from HullParameterization import Hull_Parameterization

HERE = Path(__file__).resolve().parent


class MeshParams(BaseModel):
    waterlines: int = Field(100, ge=2, description="Number of waterlines")
    points_per_waterline: int = Field(800, ge=2)
    transom: bool = True
    deck_lid: bool = True
    scale: float = Field(1.0, gt=0)


class HullParams(BaseModel):
    model_config = {"extra": "forbid"}
    """ShipD's current31 construction fields; ranges and defaults retain the corresponding historical dataset values."""
    dataset_index: int = Field(-1, ge=-1, lt=30000, title="Dataset Hull",
                               description="A hull from InputVectors_30k.npy; -1 uses the fields below")
    LOA: float = Field(10.0, gt=0.0, title="Length Overall",
                       description="Primary measure of the ship's scale in meters (10 across the dataset)")
    Lb: float = Field(0.48704600344105925, gt=0.05, lt=0.9, title="Length of Bow Taper",
                      description="Fraction of LOA")
    Ls: float = Field(0.35139932667892093, gt=0.0, lt=0.9, title="Length of Stern Taper",
                      description="Fraction of LOA")
    Bd: float = Field(0.3240192483277133, gt=0.0833, lt=0.333, title="Beam at Midship Deck",
                      description="Fraction of LOA")
    Dd: float = Field(0.06684901297075647, gt=0.05, lt=0.25, title="Depth of Hull",
                      description="Fraction of LOA")
    Bs: float = Field(0.8942892176703161, gt=0.0, lt=1.0, title="Beam at Stern Deck",
                      description="Fraction of Bd")
    WL: float = Field(0.34142121012068655, gt=0.0, lt=1.0, title="Design Draft",
                      description="Fraction of Dd")
    Bc: float = Field(0.26572517744894836, gt=0.05, lt=0.5, title="Beam at Chine",
                      description="Fraction of LOA")
    Beta: float = Field(16.435785420614273, gt=0.0, lt=45.0, title="Deadrise Angle",
                        description="Degrees")
    Rc: float = Field(0.18472362175545076, gt=0.0, lt=1.0, title="Radius of Chine",
                      description="Fraction of Bc (strictly positive)")
    Rk: float = Field(0.0003990862290144115, gt=-1.0, lt=1.0, title="Radius of Keel",
                      description="Fraction of Dd (positive or negative)")
    Abow: float = Field(-1.8561650275406922, gt=-4.0, lt=4.0, title="Bow Shape A",
                        description="BOW(z) = Az^2 + Bz + C, C solved so min(BOW(z)) = 0; scaled by Lb and Dd")
    Bbow: float = Field(1.5166344155299023, gt=-4.0, lt=4.0, title="Bow Shape B",
                        description="BOW(z) = Az^2 + Bz + C, C solved so min(BOW(z)) = 0; scaled by Lb and Dd")
    BK_z: float = Field(0.14213845905466715, gt=0.0, lt=1.0, title="Bow-Keel Intersect",
                        description="Fraction of Dd where the bow curve and the keelrise curve intersect")
    Kappa_bow: float = Field(0.7636337567766859, gt=0.0, lt=1.0, title="Start of Keelrise, Bow",
                             description="Fraction of Lb where the keel rises from z = 0 towards the bow")
    Adel_bow: float = Field(3.4918237262596694, gt=-4.0, lt=4.0, title="Bow Midship-Width Curve A",
                            description="DELTA_BOW(z) = Az^2 + Bz + C: x where the midship beam is reached along the bow taper")
    Bdel_bow: float = Field(-3.538549146494405, gt=-4.0, lt=4.0, title="Bow Midship-Width Curve B",
                            description="DELTA_BOW(z) = Az^2 + Bz + C: x where the midship beam is reached along the bow taper")
    Adrft: float = Field(1.242362591021366, gt=-4.0, lt=4.0, title="Drift Angle A",
                         description="DRIFT(z) = Az^2 + Bz + C: drift angle in degrees along BOW(z) as a function of height")
    Bdrft: float = Field(-0.9213857793078297, gt=-4.0, lt=4.0, title="Drift Angle B",
                         description="DRIFT(z) = Az^2 + Bz + C: drift angle in degrees along BOW(z) as a function of height")
    Cdrft: float = Field(24.19395370410423, gt=0.0, lt=60.0, title="Drift Angle C",
                         description="DRIFT(z) = Az^2 + Bz + C: drift angle in degrees along BOW(z) as a function of height")
    bit_EP_S: bool = Field(True, title="Lower Stern Taper", description="Stern taper below the transom: ellipse (on) or parabola (off)")
    bit_EP_T: bool = Field(True, title="Upper Stern Taper", description="Stern taper for the transom: ellipse (on) or parabola (off)")
    Atrans: float = Field(-0.41914734813071597, gt=-3.0, lt=5.0, title="Transom Slope",
                          description="Transom(z) = Az + B: the transom position between Dd and SK")
    SK_z: float = Field(0.591806951838623, gt=0.0, lt=1.0, title="Stern-Keel Intersect",
                        description="Intersection of the transom and the stern keelrise, fraction of Dd")
    Kappa_stern: float = Field(0.9729049748797866, gt=0.0, lt=1.0, title="Start of Keelrise, Stern",
                               description="Fraction of Ls where the keel rises from z = 0 towards the transom")
    Adel_stern: float = Field(-2.457254365107886, gt=-4.0, lt=4.0, title="Stern Midship-Width Curve A",
                              description="DELTA_STERN(z) = Az^2 + Bz + C: x where the midship beam is reached along the stern taper")
    Bdel_stern: float = Field(1.9997133058087724, gt=-4.0, lt=4.0, title="Stern Midship-Width Curve B",
                              description="DELTA_STERN(z) = Az^2 + Bz + C: x where the midship beam is reached along the stern taper")
    Beta_trans: float = Field(2.4011278251310686, gt=0.0, lt=60.0, title="Transom Deadrise Angle",
                              description="Degrees")
    Bc_trans: float = Field(0.14456725438136536, gt=0.0, lt=0.5, title="Beam at Transom Chine",
                            description="Fraction of LOA")
    Rc_trans: float = Field(0.06913326315069873, gt=0.0, lt=0.5, title="Transom Chine Radius",
                            description="Fraction of Bc_trans")
    Rk_trans: float = Field(0.4370949977009253, gt=-1.0, lt=1.0, title="Transom Keel Radius",
                            description="Fraction of Dd * (1 - SK)")
    mesh: MeshParams = Field(default_factory=MeshParams, title="Mesh", description="How the hull is meshed when it is shown")


class Hull:
    """The chain's value: ShipD's hull and how to mesh it, meshed once, when the app shows it."""

    def __init__(self, parameterization: Hull_Parameterization, meshing: MeshParams):
        self.parameterization, self.meshing, self._mesh = parameterization, meshing, None

    def __spatial_mesh__(self) -> tuple[np.ndarray, np.ndarray]:
        """The hull as vertices (N x 3 float32) and triangles (M x 3 int32), three vertices per triangle."""
        if self._mesh is None:
            p = self.meshing
            triangles = self.parameterization.gen_stl(NUM_WL=p.waterlines, PointsPerWL=p.points_per_waterline,
                                                      bit_AddTransom=int(p.transom), bit_AddDeckLid=int(p.deck_lid))
            vertices = (triangles.reshape(-1, 3) * p.scale).astype(np.float32)
            self._mesh = vertices, np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        return self._mesh


def hull(p: HullParams) -> Hull:
    if p.dataset_index >= 0:
        path = HERE / "InputVectors_30k.npy"
        vector = np.load(path)[p.dataset_index]
        parameterization = Hull_Parameterization.from_declared(
            vector, input_format=Hull_Parameterization.HISTORICAL_FORMAT,
            columns=Hull_Parameterization.HISTORICAL_COLUMNS,
            source=f"{path}#row={p.dataset_index}")
    else:
        fields = tuple(name for name, value in p if name not in ("dataset_index", "mesh"))
        if fields != Hull_Parameterization.INPUT_COLUMNS:
            raise ValueError("HullParams fields must match the current ShipD construction order")
        vector = np.array([float(getattr(p, name)) for name in fields])
        parameterization = Hull_Parameterization.from_declared(
            vector, input_format=Hull_Parameterization.CURRENT_FORMAT,
            columns=fields, source="SpatialAI HullParams")
    return Hull(parameterization, p.mesh)
