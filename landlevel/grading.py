"""Fit de plan de nivellement (monoplan) et calcul des volumes cut/fill.

Deux methodes de fit sont fournies :

- ``fit_plane_ols`` : moindres carres classique z = a*x + b*y + c sur tous
  les points valides. C'est la methode "Plane of Best Fit" utilisee par
  Ezigrade.
- ``fit_plane_balanced_l1`` : programme lineaire qui minimise le
  terrassement total (somme des valeurs absolues des ecarts design -
  naturel) sous une contrainte d'equilibre stricte cut = fill.

Note mathematique : un plan OLS avec intercept passe toujours par le
centroide (x_bar, y_bar, z_bar) du nuage de points, ce qui implique que la
somme des residus (design - naturel) est deja nulle. Le plan OLS est donc
*deja* equilibre cut = fill (a l'arrondi pres), a condition que toutes les
cellules valides aient le meme poids/surface. La difference reelle entre
les deux methodes n'est donc pas presence/absence d'equilibre, mais le
critere d'optimalite : OLS minimise l'erreur quadratique (sensible aux
ecarts locaux importants), alors que le plan L1 contraint minimise
directement le volume total de terre deplacee (cut + fill), ce qui est le
critere qui compte reellement pour un nivellement.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


@dataclass
class Plane:
    a: float  # dz/dx (grade transversal, direction X / Est)
    b: float  # dz/dy (grade principal, direction Y / Nord)
    c: float  # intercept
    slope_constrained: bool = False  # True si la contrainte de pente minimale etait active

    def elevation(self, x, y):
        return self.a * x + self.b * y + self.c

    def slope_pct(self) -> float:
        """Pente resultante (norme du gradient) en %."""
        return float(np.hypot(self.a, self.b) * 100.0)

    def grade_x_pct(self) -> float:
        """Grade transversal (Cross Grade, direction X) en %."""
        return self.a * 100.0

    def grade_y_pct(self) -> float:
        """Grade principal (Main Grade, direction Y) en %."""
        return self.b * 100.0

    def bearing_deg(self) -> float:
        """Azimut compas (0=Nord, sens horaire) de la direction de plus
        forte pente descendante du plan fitte."""
        return float(np.degrees(np.arctan2(-self.a, -self.b)) % 360.0)


def fit_plane_ols(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Plane:
    """Plan z = a*x + b*y + c par moindres carres classiques (non contraint)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    return Plane(a=float(a), b=float(b), c=float(c))


def _solve_balanced_l1(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        ratio: float = 1.0, slope_dir=None, min_slope: float = 0.0) -> Plane:
    """Plan minimisant le terrassement total sous contrainte d'equilibre cut/fill.

    Si `slope_dir` = (ux, uy) (vecteur unitaire) est fourni, ajoute la
    contrainte lineaire  a*ux + b*uy >= min_slope  (pente d'au moins
    `min_slope` dans cette direction). Voir fit_plane_balanced_l1.

    Variables du LP : [a, b, c, t_1..t_n, T] ou t_i >= |design_i - z_i| et
    T = sum(t_i) (variable auxiliaire, introduite seulement pour exprimer
    la contrainte de ratio ci-dessous sous forme lineaire) :

    minimiser   sum(t_i)
    sous        t_i - (a*x_i + b*y_i + c - z_i) >= 0
                t_i + (a*x_i + b*y_i + c - z_i) >= 0
                T - sum(t_i) = 0
                (1+ratio)*sum(a*x_i + b*y_i + c - z_i) - (1-ratio)*T = 0

    Justification de la contrainte de ratio : en notant S = sum(design_i -
    z_i) (lineaire) et T = sum(t_i) = sum(|design_i - z_i|), on a toujours
    remblai = (S+T)/2 et deblai = (T-S)/2 (car S = remblai - deblai et T =
    remblai + deblai). Imposer deblai = ratio * remblai revient exactement
    a (1+ratio)*S - (1-ratio)*T = 0. Pour ratio=1, ca redonne S=0,
    l'equilibre strict cut=fill.

    ratio=1.0 (defaut) : equilibre strict cut = fill.
    ratio dans [1.1, 1.5] : ratio de foisonnement/tassement typique (FAO ;
    aussi un champ "cut/fill ratio" natif dans Ezigrade, cf. doc section
    design) — le deblai necessaire depasse le remblai compacte car le sol
    remanie change de volume.

    Cette formulation garde la structure a 2 blocs d'inegalites (rapide) et
    n'ajoute qu'UNE variable (T) et deux egalites pour geler le ratio,
    plutot que de doubler les n variables (decomposition pos_i/neg_i,
    mesuree ~8x plus lente). Pour ratio=1.0 le coefficient de T dans la
    contrainte de ratio s'annule exactement ((1-ratio)=0) et le solveur
    retrouve quasi la vitesse du cas non-generalise (~40-45s sur ~50k
    points). Des que ratio != 1.0, T est reellement couple au reste du
    systeme et HiGHS-IPM est nettement plus lent sur ce jeu de donnees
    (mesure : ~330s a ratio=1.25 sur ~50k points, contre ~45s a ratio=1.0)
    — prevoir ce cout pour un fit pleine resolution ; sur une zone de
    quelques milliers de points (cas typique apres decoupage par zone), le
    surcout reste negligeable en temps absolu.

    Les contraintes sont assemblees en matrices creuses (sparse) car le
    nombre de points peut atteindre plusieurs dizaines de milliers.
    """
    if ratio <= 0:
        raise ValueError(f"ratio doit etre > 0, recu {ratio}")

    n = len(x)
    n_vars = 4 + n  # [a, b, c, t_1..t_n, T]
    T_IDX = 3 + n

    c_obj = np.zeros(n_vars)
    c_obj[3:3 + n] = 1.0

    idx = np.arange(n)

    # Bloc 1 (inegalite) : a*x_i + b*y_i + c - t_i <= z_i
    rows1_r = np.concatenate([np.repeat(idx, 3), idx])
    rows1_c = np.concatenate([np.tile([0, 1, 2], n), 3 + idx])
    rows1_v = np.concatenate([
        np.column_stack([x, y, np.ones(n)]).ravel(),
        -np.ones(n),
    ])
    block1 = sp.csr_matrix((rows1_v, (rows1_r, rows1_c)), shape=(n, n_vars))
    b1 = z

    # Bloc 2 (inegalite) : -a*x_i - b*y_i - c - t_i <= -z_i
    rows2_v = np.concatenate([
        np.column_stack([-x, -y, -np.ones(n)]).ravel(),
        -np.ones(n),
    ])
    block2 = sp.csr_matrix((rows2_v, (rows1_r, rows1_c)), shape=(n, n_vars))
    b2 = -z

    A_ub = sp.vstack([block1, block2], format="csr")
    b_ub = np.concatenate([b1, b2])

    if slope_dir is not None and min_slope > 0:
        # a*ux + b*uy >= min_slope  <=>  -ux*a - uy*b <= -min_slope
        ux, uy = slope_dir
        slope_row = sp.csr_matrix(
            (np.array([-ux, -uy]), ([0, 0], [0, 1])), shape=(1, n_vars),
        )
        A_ub = sp.vstack([A_ub, slope_row], format="csr")
        b_ub = np.concatenate([b_ub, [-min_slope]])

    # Egalite 1 : T - sum(t_i) = 0
    A_T = sp.csr_matrix(
        (
            np.concatenate([-np.ones(n), [1.0]]),
            (np.zeros(n + 1, dtype=int), np.concatenate([3 + idx, [T_IDX]])),
        ),
        shape=(1, n_vars),
    )
    b_T = np.array([0.0])

    # Egalite 2 : (1+ratio)*S - (1-ratio)*T = 0, S = sum(a*x_i+b*y_i+c-z_i)
    A_ratio = sp.csr_matrix(
        (
            np.array([(1 + ratio) * x.sum(), (1 + ratio) * y.sum(), (1 + ratio) * n,
                      -(1 - ratio)]),
            ([0, 0, 0, 0], [0, 1, 2, T_IDX]),
        ),
        shape=(1, n_vars),
    )
    b_ratio = np.array([(1 + ratio) * z.sum()])

    A_eq = sp.vstack([A_T, A_ratio], format="csr")
    b_eq = np.concatenate([b_T, b_ratio])

    bounds = [(None, None)] * 3 + [(0, None)] * n + [(None, None)]

    # "highs-ipm" (point interieur) est nettement plus rapide que le simplexe
    # ("highs"/"highs-ds") sur ce type de LP (~4x plus vite mesure sur ce
    # jeu de donnees, 38s vs 167s pour ~50k points).
    res = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs-ipm",
    )
    if not res.success:
        raise RuntimeError(f"echec du fit LP contraint : {res.message}")

    a, b, c = res.x[:3]
    return Plane(a=float(a), b=float(b), c=float(c))


def fit_plane_balanced_l1(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           ratio: float = 1.0, min_slope: float = 0.0) -> Plane:
    """Plan de nivellement L1 equilibre (cut/fill), avec pente minimale optionnelle.

    ratio : ratio cut/fill cible (1.0 = equilibre strict ; 1.1-1.5 = foisonnement).
    min_slope : pente resultante minimale, en fraction (0.0015 = 0,15 %).
        Sans pente minimale, le solveur peut produire un plan quasi plat
        (moins de terrassement), ce qui ne garantit pas l'egouttement de
        surface : les guides MAPAQ demandent au moins 0,1 a 0,15 % de pente
        finale pour eviter les zones a pente nulle et l'accumulation d'eau.

    "|gradient| >= min_slope" n'est pas convexe (complement d'un disque),
    donc pas exprimable en LP. On procede en deux temps :
      1. resolution sans contrainte de pente ; si |gradient| >= min_slope,
         c'est termine (le plus frequent, un seul LP) ;
      2. sinon on fige la direction de plus forte pente u = gradient/|gradient|
         du plan non contraint (a defaut, celle du plan OLS du terrain) et on
         resout le LP avec la contrainte lineaire  gradient . u >= min_slope.
         Cela conserve le sens d'ecoulement et garantit |gradient| >= min_slope.
    Le plan retourne porte `slope_constrained=True` dans le second cas.
    """
    plane = _solve_balanced_l1(x, y, z, ratio=ratio)
    if min_slope <= 0:
        return plane

    slope = float(np.hypot(plane.a, plane.b))
    if slope >= min_slope:
        return plane

    if slope > 1e-9:
        direction = (plane.a / slope, plane.b / slope)
    else:
        ols = fit_plane_ols(x, y, z)
        ols_slope = float(np.hypot(ols.a, ols.b))
        direction = (ols.a / ols_slope, ols.b / ols_slope) if ols_slope > 1e-9 else (0.0, -1.0)

    constrained = _solve_balanced_l1(x, y, z, ratio=ratio, slope_dir=direction, min_slope=min_slope)
    constrained.slope_constrained = True
    return constrained


def compute_volumes(z_natural: np.ndarray, z_design: np.ndarray, cell_area: float):
    """Calcule cut/fill (m3) avec la convention Ezigrade : cut negatif, fill positif.

    Retourne (cut_volume, fill_volume, net_imbalance).
    """
    diff = z_design - z_natural  # >0 => remblai (fill), <0 => deblai (cut)
    fill_volume = float(np.sum(diff[diff > 0]) * cell_area)
    cut_volume = float(np.sum(diff[diff < 0]) * cell_area)
    net_imbalance = cut_volume + fill_volume
    return cut_volume, fill_volume, net_imbalance
