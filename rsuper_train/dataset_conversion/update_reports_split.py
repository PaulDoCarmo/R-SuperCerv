#!/usr/bin/env python3
"""update_reports_split.py -- Étend le split rapports avec un NOUVEAU lot, SANS reshuffle.

make_reports_split.py (seed=0) régénère TOUT -> re-lancé sur un dataset agrandi il change le
held-out (casse la comparabilité). Ici on met à jour de façon INCRÉMENTALE :
  - patients de TEST gelés : le held-out garde EXACTEMENT les mêmes patients (comparabilité).
  - scan nouveau dont le patient est déjà en test  -> test  (sinon fuite).
  - scan nouveau dont le patient est déjà en train -> train.
  - patient entièrement nouveau -> train (on ne grossit pas le held-out avec des patients frais).
Garantit : aucune fuite patient, baseline0 ⊂ all. Vues produites = reports_all + reports_baseline0.

Usage :
  python update_reports_split.py --splits_dir $D/splits --new_ids $D/report_extraction/reports_newbatch.csv
"""
import argparse, csv, os, re, pandas as pd


def read_ids(path):
    d = pd.read_csv(path, header=None)
    ids = [str(x) for x in d.iloc[:, 0]]
    if ids and ids[0].lower() in ("id", "bdmap_id", "ids"):
        ids = ids[1:]
    return ids


def base(stem):
    m = re.match(r"^(.*)_([01])$", stem)
    return m.group(1) if m else stem


def write_ids(path, ids):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["BDMAP_ID"])
        for i in sorted(ids):
            w.writerow([i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--new_ids", required=True, help="CSV avec colonne ID des nouveaux stems <patient>_<0|1>")
    a = ap.parse_args()
    S = a.splits_dir

    all_tr = read_ids(f"{S}/reports_all_train_ids.csv")
    all_te = read_ids(f"{S}/reports_all_test_ids.csv")
    test_pat = {base(i) for i in all_te}
    train_pat = {base(i) for i in all_tr}
    assert not (test_pat & train_pat), "fuite patient PRÉ-EXISTANTE dans le split !"

    new = read_ids(a.new_ids) if a.new_ids.endswith(".csv") and "BDMAP" not in open(a.new_ids).readline() \
        else [str(x) for x in pd.read_csv(a.new_ids)["ID"]]
    new = [s for s in new if s not in set(all_tr) | set(all_te)]      # net-nouveau seulement

    to_test, to_train, n_newpat = [], [], 0
    for s in new:
        p = base(s)
        if p in test_pat:
            to_test.append(s)
        elif p in train_pat:
            to_train.append(s)
        else:                                                        # patient nouveau -> train (test gelé)
            to_train.append(s); n_newpat += 1

    all_te2 = sorted(set(all_te) | set(to_test))
    all_tr2 = sorted(set(all_tr) | set(to_train))
    b0_te = [s for s in all_te2 if s.endswith("_0")]
    b0_tr = [s for s in all_tr2 if s.endswith("_0")]

    # vérifs
    pte = {base(i) for i in all_te2}; ptr = {base(i) for i in all_tr2}
    assert not (pte & ptr), "fuite patient après update !"
    assert pte == test_pat, "les patients de TEST ne doivent PAS changer (held-out gelé) !"
    assert set(b0_te) <= set(all_te2) and set(b0_tr) <= set(all_tr2)

    write_ids(f"{S}/reports_all_test_ids.csv", all_te2)
    write_ids(f"{S}/reports_all_train_ids.csv", all_tr2)
    write_ids(f"{S}/reports_baseline0_test_ids.csv", b0_te)
    write_ids(f"{S}/reports_baseline0_train_ids.csv", b0_tr)

    print(f"nouveaux routés : test+={len(to_test)}  train+={len(to_train)} (dont {n_newpat} patients neufs -> train)")
    print(f"reports_all       : train {len(all_tr)}->{len(all_tr2)}  test {len(all_te)}->{len(all_te2)}  "
          f"(patients test={len(pte)} INCHANGÉS, train={len(ptr)})")
    print(f"reports_baseline0 : train={len(b0_tr)}  test={len(b0_te)}")
    print(f"test_frac scans={len(all_te2)/(len(all_tr2)+len(all_te2)):.3f} | OK: 0 fuite, held-out gelé, baseline0 ⊂ all")


if __name__ == "__main__":
    main()
