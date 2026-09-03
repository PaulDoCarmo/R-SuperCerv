#!/usr/bin/env python3
"""
Loss de PRÉSENCE IVH (détection faible-supervisée par rapports)
================================================================

But
---
Améliorer la DÉTECTION d'IVH du MODÈLE SEUL. Le rapport radiologique n'est PAS
disponible au test : il ne sert que de signal d'entraînement faible sur les exams CHUM
(rapport-only). La cible est la sensibilité (rattraper les faux négatifs du modèle), pas
le Dice — l'IVH est déjà à Dice 0.53-0.77 et non saturé.

Ce que cette loss N'EST PAS
---------------------------
Ce n'est PAS un pseudo-masque ni une Volume/Ball loss. On ne dessine pas l'IVH.
On pose une contrainte de PRÉSENCE poolée : « il y a (ou non) de l'IVH dans la région
ventriculaire », dérivée du flag rapport (type=IVH), sans jamais localiser au voxel.

Le mécanisme (3 termes), décidés par 4 études data-driven (cf. plus bas)
-----------------------------------------------------------------------
1) SCORE DE PRÉSENCE par LogSumExp pooling (forme ChestX-ray8 / Pinheiro-Collobert),
   γ≈10, sur la région de pooling R_pool :

       s_i = (1/γ) · [ logsumexp_{x∈R_pool} (γ · p_IVH(x))  −  log|R_pool| ]  ∈ [0,1]

   Le gradient est softmax(γ·p) sur la région : il va vers les voxels les plus "chauds".
   Implémentation STABILISÉE (torch.logsumexp), jamais la forme naïve Σ exp.

2) HINGE ASYMÉTRIQUE (type Kervadec), y_i = flag IVH du rapport (par SCAN), τ⁺ > τ⁻ :

       L_pres(i) = [max(0, τ⁺ − s_i)]²                si y_i = 1   (pousser vers le haut)
                 = λ⁻ · [max(0, s_i − τ⁻)]²           si y_i = 0   (supprimer, affaibli)

   λ⁻≈0.9 encode les ~14% de faux négatifs du rapport : on affaiblit le côté négatif
   pour ne pas apprendre à EFFACER une vraie IVH que le rapport a manquée
   (4/29 cas masque+/rapport− observés ; 3/4 sont déjà bien détectés par le modèle).

3) SUPPRESSION DE FOND sur un scope large et PLEIN R_supp = ventricule-dilaté-15mm
   (containment IVH 100%), INDÉPENDANTE de y_i :

       L_bkg = − (1/|¬R_supp|) · Σ_{x∉R_supp} log(1 − p_IVH(x))

   Empêche le modèle de peindre de l'IVH loin du système ventriculaire.

Objectif global (assemblé dans calculate_loss, pas ici) :

    L = L_CE+Dice^{RSNA, tous canaux}  +  λ · ( L_pres + β · L_bkg )^{CHUM, canal IVH}

avec λ≈0.1 en RAMPE depuis 0, batchs équilibrés masques/rapports et positifs/négatifs,
la seg-loss masquée RSNA TOUJOURS allumée (ancre anti-dérive).

Décisions figées par les études (sur 29 IVH+ / 36 IVH− CHUM, modèle X25) — TRAÇABILITÉ
------------------------------------------------------------------------------------
· R_pool = ventricule TotalSeg dilaté 10 mm. Le ventricule BRUT échoue (containment 5%,
  noyé par le sang) ; dilaté 10 mm → 99%. Le ventricule n'est JAMAIS absent (médiane 33 mL,
  min 15.7 mL, 0 cas vide) car TotalSeg le segmente par anatomie, pas par intensité.
· PAS de purification p_ICH. Étude : le raccourci ICH (le LSE saturerait depuis l'ICH
  hyperdense) N'EXISTE PAS en pratique — le canal IVH du modèle ne peint pas d'IVH sur
  l'ICH (p_IVH≈0.006 sur ICH ; masse LSE venant de l'ICH ≤1%, jamais >50% sur 29 cas).
  Une pondération douce (1−p_ICH) ne dominait pas la géométrie nue et PÉNALISAIT les FN
  (atténue l'IVH aux bords). Donc géométrie pure → pas de circularité, pas de .detach().
· γ=10 : sépare le mieux les histogrammes s_i (IVH+ médian 0.51 vs IVH− 0.11). γ plus haut
  compresse tout vers 1 (le LSE→max, un négatif a toujours un voxel tiède).
· τ⁺=0.42 (p20 des positifs), τ⁻=0.30 (p92 des négatifs) — bande valide (τ⁺>τ⁻).
  Un seul τ⁺ : unilatéral (0.475) ≈ médian/bilatéral (0.529) à γ=10, pas de seuil conditionnel.
· LIMITE documentée : un FN froid ET rapport-négatif est IRRÉCUPÉRABLE (1 cas, DA9555F8,
  IVH 0.11 mL non mentionnée par le radiologue). Ni s_i ni le rapport ne le signalent.
  Le "seed d'intensité" reste en réserve mais non activé (1 cas sub-0.11 mL ne le justifie pas).
· RÉSERVE : n petit (calibration 29+/36−, gain mesurable sur ~5 FN). Preuve de faisabilité
  du mécanisme, PAS résultat final. Augmenter n (par-lésion, puis élargir CHUM) avant conclusion.

Interface (probabilités, PAS logits)
------------------------------------
Toutes les fonctions prennent p_ivh = sigmoid(logit_IVH) ∈ [0,1] pour que s_i ∈ [0,1].
r_pool / r_supp sont des masques (bool ou float 0/1) de MÊME forme spatiale que p_ivh.
Formes acceptées : (B, *spatial) ou (B, 1, *spatial) — l'axe canal singleton est aplati.
"""
import torch
import torch.nn.functional as F


# Valeurs par défaut = calibration figée (études 1-4). Surchargeables via args au training.
DEFAULTS = dict(gamma=10.0, tau_pos=0.42, tau_neg=0.30, lambda_neg=0.9, beta=1.0)


def _flatten_bt(x):
    """(B, *spatial) ou (B, 1, *spatial) -> (B, N). Aplati l'axe canal singleton éventuel."""
    if x.dim() >= 3 and x.shape[1] == 1:
        x = x[:, 0]
    return x.reshape(x.shape[0], -1)


def presence_score(p_ivh, r_pool, gamma=DEFAULTS["gamma"]):
    """Score de présence par LogSumExp pooling STABILISÉ (max-shift), par item du batch.

        s_i = p* + (1/γ)·log( (1/|R_pool|)·Σ_{x∈R_pool} exp(γ(p_x − p*)) ),   p* = max_x p_x

    Forme mathématiquement identique à (1/γ)·[logsumexp(γp) − log|R_pool|] mais sans jamais
    d'overflow : après soustraction du max, le plus grand terme vaut e^0=1, les autres ∈(0,1].
    On délègue le max-shift à torch.logsumexp (qui le fait en interne) plutôt que de coder la
    forme naïve.  s_i ∈ [0,1] (car p ∈ [0,1]) ; sortir de [0,1] signalerait une erreur de norm.

    Masquage par INDEX BOOLÉEN p[r] (et NON multiplication par masque 0/1) : seuls les voxels
    de R_pool entrent dans le pool. Une multiplication laisserait les voxels hors-ancre comme
    des "p=0" -> gonfle |R_pool| et refroidit artificiellement s_i. L'indexation les retire
    vraiment ET préserve le gradient (il ne va qu'aux voxels d'ancre).

    Args:
        p_ivh : (B,*spatial) ou (B,1,*spatial), probabilités IVH ∈ [0,1].
        r_pool: même forme, masque de pooling (bool/float ; seuillé >0.5).
        gamma : netteté (γ→∞ => max ; γ→0 => moyenne).
    Returns:
        s : (B,) ∈ [0,1].
    """
    p = _flatten_bt(p_ivh)
    r = _flatten_bt(r_pool) > 0.5
    outs = []
    for b in range(p.shape[0]):
        vals = p[b][r[b]]                                  # index booléen -> (|R_pool|,), gradient préservé
        if vals.numel() == 0:                              # pool vide (jamais en pratique : ventricule toujours présent)
            outs.append(p.new_zeros(())); continue
        s = (torch.logsumexp(gamma * vals, dim=0)          # max-shift interne
             - torch.log(vals.new_tensor(float(vals.numel())))) / gamma
        outs.append(s)
    return torch.stack(outs).clamp(0.0, 1.0)


def presence_hinge_loss(s, y_ivh, tau_pos=DEFAULTS["tau_pos"], tau_neg=DEFAULTS["tau_neg"],
                        lambda_neg=DEFAULTS["lambda_neg"], reduction="mean"):
    """Hinge asymétrique de Kervadec sur le score de présence.

        y=1 : [max(0, τ⁺ − s)]²           (pousse s vers le haut si sous τ⁺)
        y=0 : λ⁻ · [max(0, s − τ⁻)]²      (pousse s vers le bas si au-dessus τ⁻, AFFAIBLI)

    Zone morte [τ⁻, τ⁺] : aucun gradient si s y est déjà (positif au-dessus de τ⁺, négatif
    sous τ⁻ -> loss nulle). λ⁻<1 protège les vraies IVH manquées par le rapport.

    Args:
        s     : (B,) score de présence ∈ [0,1].
        y_ivh : (B,) flag rapport IVH ∈ {0,1} (par SCAN, matching exact du cid).
        reduction : 'mean' | 'none'.
    Returns:
        scalaire (mean) ou (B,) (none).
    """
    pos = torch.clamp(tau_pos - s, min=0.0) ** 2
    neg = lambda_neg * torch.clamp(s - tau_neg, min=0.0) ** 2
    loss = torch.where(y_ivh > 0.5, pos, neg)
    return loss.mean() if reduction == "mean" else loss


def background_suppression_loss(p_ivh, r_supp, eps=1e-6):
    """Suppression de fond hors R_supp (ventricule-dilaté-15mm), indépendante de y_i.

        L_bkg = − (1/|¬R_supp|) · Σ_{x∉R_supp} log(1 − p_IVH(x))     (BCE cible 0)

    R_supp doit rester PLEIN et LARGE (containment IVH 100%) : on ne supprime QUE loin
    du système ventriculaire, jamais dans la lumière où l'IVH peut être.
    """
    p = _flatten_bt(p_ivh)
    bg = _flatten_bt(r_supp) <= 0.5                          # hors R_supp
    logq = torch.log((1.0 - p).clamp(min=eps))
    num = -(logq * bg).sum(dim=1)
    den = bg.sum(dim=1).clamp(min=1).float()
    return (num / den).mean()


def ivh_presence_loss(p_ivh, r_pool, r_supp, y_ivh,
                      gamma=DEFAULTS["gamma"], tau_pos=DEFAULTS["tau_pos"],
                      tau_neg=DEFAULTS["tau_neg"], lambda_neg=DEFAULTS["lambda_neg"],
                      beta=DEFAULTS["beta"]):
    """Loss de présence IVH complète (hinge + β·suppression de fond).

    Le poids global λ (et sa rampe) est appliqué DANS calculate_loss, pas ici.

    Args:
        p_ivh : (B,*spatial) proba IVH ∈ [0,1] = sigmoid(logit du canal ivh_lesion).
        r_pool: (B,*spatial) ventricule-dilaté-10mm (pooling).
        r_supp: (B,*spatial) ventricule-dilaté-15mm (suppression de fond, plein).
        y_ivh : (B,) flag rapport IVH ∈ {0,1} par scan.
    Returns:
        dict {'ivh_presence','ivh_bkg','ivh_s_mean'(detaché, monitoring)}.
    """
    s = presence_score(p_ivh, r_pool, gamma)
    l_pres = presence_hinge_loss(s, y_ivh, tau_pos, tau_neg, lambda_neg)
    l_bkg = background_suppression_loss(p_ivh, r_supp)
    return {"ivh_presence": l_pres, "ivh_bkg": beta * l_bkg, "ivh_s_mean": s.mean().detach()}


# ----------------------------------------------------------------------------------------
# Auto-test : vérifie les propriétés-clés (présence, asymétrie, zone morte, suppression,
# sûreté-raccourci ICH hors R_pool). `python -m training.ivh_presence_loss`
# ----------------------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    B, D, H, W = 4, 16, 16, 16
    p = torch.zeros(B, D, H, W)
    r_pool = torch.zeros(B, D, H, W); r_pool[:, 4:12, 4:12, 4:12] = 1     # ventricule "dilaté"
    r_supp = torch.zeros(B, D, H, W); r_supp[:, 2:14, 2:14, 2:14] = 1     # plus large
    # item 0 : IVH+ bien détectée (blob chaud DANS R_pool)
    p[0, 6:9, 6:9, 6:9] = 0.95
    # item 1 : IVH+ FROIDE (rien de chaud -> FN) : s bas, le hinge doit pousser
    p[1] = 0.02
    # item 2 : IVH- avec un blob chaud DANS R_pool (le hinge négatif doit supprimer)
    p[2, 6:9, 6:9, 6:9] = 0.9
    # item 3 : blob chaud type ICH HORS R_pool (sûreté-raccourci : ne doit PAS lever s)
    p[3, 0:3, 0:3, 0:3] = 0.99
    y = torch.tensor([1., 1., 0., 1.])

    s = presence_score(p, r_pool, gamma=10.0)
    print("scores s_i :", [f"{v:.3f}" for v in s.tolist()])
    assert s[0] > 0.5,  "IVH+ détectée -> s haut"
    assert s[1] < 0.1,  "IVH+ froide -> s bas (c'est un FN, rattrapé par le flag rapport)"
    assert s[2] > 0.5,  "blob chaud dans R_pool -> s haut"
    assert s[3] < 0.1,  "blob chaud HORS R_pool -> s bas (sûreté-raccourci OK)"

    # hinge : zone morte + asymétrie
    l_dead = presence_hinge_loss(torch.tensor([0.9]), torch.tensor([1.]))   # y=1, s>τ+ -> 0
    l_push = presence_hinge_loss(torch.tensor([0.05]), torch.tensor([1.]))  # y=1, s<τ+ -> >0
    l_supp = presence_hinge_loss(torch.tensor([0.9]), torch.tensor([0.]))   # y=0, s>τ- -> >0
    l_ok   = presence_hinge_loss(torch.tensor([0.05]), torch.tensor([0.]))  # y=0, s<τ- -> 0
    print(f"hinge  push(y1,s.05)={l_push:.3f}  dead(y1,s.9)={l_dead:.3f}  "
          f"supp(y0,s.9)={l_supp:.3f}  ok(y0,s.05)={l_ok:.3f}")
    assert l_dead == 0 and l_ok == 0,   "zone morte : pas de gradient si déjà du bon côté"
    assert l_push > 0 and l_supp > 0,   "pousse les FN et supprime les FP"
    # asymétrie λ⁻ : à écart égal, le côté négatif pèse moins
    d = 0.2
    lp = presence_hinge_loss(torch.tensor([DEFAULTS['tau_pos'] - d]), torch.tensor([1.]))
    ln = presence_hinge_loss(torch.tensor([DEFAULTS['tau_neg'] + d]), torch.tensor([0.]))
    print(f"asymétrie : pos={lp:.4f} vs neg={ln:.4f} (neg = λ⁻·pos = {DEFAULTS['lambda_neg']}·pos)")
    assert abs(ln - DEFAULTS['lambda_neg'] * lp) < 1e-5, "λ⁻ affaiblit bien le côté négatif"

    # gradient : pousse-t-il les bons voxels ? (backward sur un FN)
    p1 = p.clone().requires_grad_(True)
    out = ivh_presence_loss(p1, r_pool, r_supp, y)
    (out["ivh_presence"] + out["ivh_bkg"]).backward()
    g = p1.grad[1]                                          # item 1 = FN froid, y=1
    # le gradient (descente) doit être négatif (=> augmenter p) DANS R_pool, ~0 hors
    gin = g[r_pool[1] > 0.5].mean(); gout = g[r_pool[1] <= 0.5].abs().mean()
    print(f"FN froid : grad moyen dans R_pool={gin:.2e} (<0 => pousse p vers le haut), hors={gout:.2e}")
    assert gin < 0, "le hinge positif pousse p_IVH vers le haut dans R_pool sur un FN"
    # suppression de fond : blob ICH hors R_supp (item 3, coin) doit être pénalisé
    assert out["ivh_bkg"] > 0, "suppression de fond active"
    print("\n[OK] tous les tests passent — la loss de présence se comporte comme spécifié.")


if __name__ == "__main__":
    _selftest()
