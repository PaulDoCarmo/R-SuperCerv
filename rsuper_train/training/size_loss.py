"""Perte de contrainte de taille pour segmentation faiblement supervisee.

Reprise de Kervadec et al., "Constrained-CNN losses for weakly supervised
segmentation" (Medical Image Analysis, 2019), Eq. 2, SIMPLIFIEE pour notre cas :
  - SANS le terme d'attache aux donnees H(S) (l'ex cross-entropy partielle) :
    ici la seule supervision est la contrainte de taille issue du rapport ;
  - la bande [a, b] est definie par une TOLERANCE relative autour du volume-cible V
    (au lieu de bornes a, b fixes) :
        a = (1 - tol) * V ,   b = (1 + tol) * V .

Perte (par item du batch et par canal) :
    V_S    = somme des probabilites predites dans la region (soft, apres sigmoid)
    C(V_S) = (V_S - a)^2   si V_S < a
             (V_S - b)^2   si V_S > b
             0             sinon
    L      = lambda_size * C(V_S)

Volontairement INDEPENDANTE du code R-Super (volume_loss_basic / ball_loss) :
module autonome, testable, reutilisable.

Echelle : C est en (voxels)^2 -> croit avec la taille de la lesion. Deux options :
  - normalize=False (defaut, fidele au papier) : ponderer via lambda_size (petit) ;
  - normalize=True : divise C par (V_S + V)^2 -> perte BORNEE [0, ~1) des deux cotes,
    scale-invariante entre petites et grosses lesions (recommande ; ÷V^2 seul laissait
    exploser la sur-prediction et dominait la seg).

Tolerance par defaut = 0.4 : choix DATA-DRIVEN. Notre volume rapport (ABC/2 debiaise,
C=equal_A) a un ratio est/reel de dispersion p25-p75 ~ [0.87, 1.45] ; une bande a
+/-40 % "tolere" le vrai volume dans ~73 % des cas (contre 44 % a +/-20 %).
"""
from __future__ import annotations
import torch


def predicted_volume(prob: torch.Tensor, region_mask: torch.Tensor | None = None,
                     spatial_dims: tuple[int, ...] | None = None) -> torch.Tensor:
    """Volume soft V_S = somme des probabilites (dans region_mask si fourni).

    prob : (B, C, *spatial) probabilites [0, 1] (appliquer sigmoid en amont).
    region_mask : meme forme, 0/1 ; None = tout le volume.
    Retourne : (B, C).
    """
    if spatial_dims is None:
        spatial_dims = tuple(range(2, prob.ndim))
    if region_mask is not None:
        prob = prob * region_mask
    return prob.sum(dim=spatial_dims)


def size_constraint(v_s: torch.Tensor, target_volume: torch.Tensor,
                    tolerance: float = 0.4, normalize: bool = False,
                    eps: float = 1.0) -> torch.Tensor:
    """C(V_S) quadratique a bande de tolerance (Eq. 2 simplifiee).

    v_s, target_volume : (B, C) en voxels. tolerance : tol relatif.
    Retourne C(V_S) par (B, C) (>= 0).
    """
    a = (1.0 - tolerance) * target_volume
    b = (1.0 + tolerance) * target_volume
    below = torch.clamp(a - v_s, min=0.0)     # > 0 ssi V_S < a
    above = torch.clamp(v_s - b, min=0.0)     # > 0 ssi V_S > b
    c = below * below + above * above         # les deux termes sont mutuellement exclusifs
    if normalize:
        # normalisation par (V_S + V)^2 : borne C dans [0, ~1) des DEUX cotes.
        # (÷V^2 seul laissait exploser la sur-prediction V_S >> V -> instable.)
        c = c / ((v_s + target_volume) * (v_s + target_volume) + eps)
    return c


def size_constraint_ab(v_s: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                       normalize: bool = True, eps: float = 1.0) -> torch.Tensor:
    """C(V_S) avec bornes EXPLICITES [a, b] (au lieu de V±tol). Version report-only :
        C = [max(0, a - V_S)]² (sous a)  +  [max(0, V_S - b)]² (au-dessus b).
    normalize : chaque terme divisé par (V_S+a)² resp. (V_S+b)² -> borné [0,~1), et le terme
    HAUT s'annule proprement si b=+inf (floor asymétrique) sans faire disparaître le terme BAS.
    a, b, v_s : (B, C) en voxels. b peut être très grand (ex. 1e9) pour 'pas de borne haute'."""
    below = torch.clamp(a - v_s, min=0.0)
    above = torch.clamp(v_s - b, min=0.0)
    if normalize:
        c = below * below / ((v_s + a) * (v_s + a) + eps) + above * above / ((v_s + b) * (v_s + b) + eps)
    else:
        c = below * below + above * above
    return c


def volume_size_loss_ab(prob: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                        region_mask: torch.Tensor | None = None, lambda_size: float = 1.0,
                        normalize: bool = True, valid: torch.Tensor | None = None,
                        reduction: str = "mean") -> torch.Tensor:
    """Perte de taille à bornes explicites [a,b] : lambda_size * C_ab(V_S). Voir size_constraint_ab.
    prob (B,C,*spatial) ; a,b (B,C) voxels ; region_mask où compter V_S ; valid (B,C) 0/1."""
    v_s = predicted_volume(prob, region_mask)
    c = size_constraint_ab(v_s, a, b, normalize=normalize)
    loss = lambda_size * c
    if valid is not None:
        loss = loss * valid
        denom = valid.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(loss.numel()), device=loss.device)
    if reduction == "mean":
        return loss.sum() / denom
    if reduction == "sum":
        return loss.sum()
    return loss


def volume_size_loss(prob: torch.Tensor, target_volume: torch.Tensor,
                     tolerance: float = 0.4, region_mask: torch.Tensor | None = None,
                     lambda_size: float = 1.0, normalize: bool = False,
                     valid: torch.Tensor | None = None,
                     reduction: str = "mean") -> torch.Tensor:
    """Perte de taille complete : lambda_size * C( V_S ).

    prob          : (B, C, *spatial) probabilites [0,1].
    target_volume : (B, C) volume cible en voxels (0 => contrainte "pas de lesion" :
                    pousse V_S -> 0). Utiliser `valid` pour desactiver les canaux
                    sans cible (taille inconnue).
    region_mask   : (B, C, *spatial) 0/1 ou compter V_S (defaut : tout).
    valid         : (B, C) 0/1 ; 1 = canal supervise. None = tous.
    reduction     : 'mean' | 'sum' | 'none'.
    """
    v_s = predicted_volume(prob, region_mask)
    c = size_constraint(v_s, target_volume, tolerance=tolerance, normalize=normalize)
    loss = lambda_size * c
    if valid is not None:
        loss = loss * valid
        denom = valid.sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(loss.numel()), device=loss.device)
    if reduction == "mean":
        return loss.sum() / denom
    if reduction == "sum":
        return loss.sum()
    return loss  # 'none' -> (B, C)


if __name__ == "__main__":
    # Auto-test : verifie la forme "vallee plate" de C et le comportement de la loss.
    B, C, S = 2, 1, 32
    torch.manual_seed(0)
    V = torch.tensor([[1000.0], [4000.0]])            # volumes cibles (voxels)
    for scale, tag in [(1.0, "V_S = V (dans la bande)"),
                       (0.5, "V_S = 0.5V (sous a)"),
                       (1.8, "V_S = 1.8V (au-dessus b)")]:
        prob = torch.zeros(B, C, S, S, S)
        # place ~scale*V voxels a proba 1 (approx via un bloc)
        for b in range(B):
            n = int(scale * V[b, 0].item()); side = round(n ** (1 / 3))
            prob[b, 0, :side, :side, :side] = 1.0
        vs = predicted_volume(prob)
        loss = volume_size_loss(prob, V, tolerance=0.4, reduction="none")
        print(f"{tag:26s} V_S={vs.squeeze().tolist()} target={V.squeeze().tolist()} "
              f"C={loss.squeeze().tolist()}")
    # tolerance : dans la bande -> 0
    vs = torch.tensor([[1100.0]]); tv = torch.tensor([[1000.0]])
    assert size_constraint(vs, tv, tolerance=0.4).item() == 0.0, "1100 doit etre dans [600,1400]"
    assert size_constraint(torch.tensor([[500.0]]), tv, 0.4).item() > 0, "500 < a=600 -> penalise"
    print("OK auto-test size_loss")
