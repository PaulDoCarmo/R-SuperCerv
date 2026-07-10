#!/usr/bin/env python3
"""make_reports_split.py -- Splits train/test PATIENT-LEVEL du dataset-RAPPORTS (UFO).

Les cas rapports sont <patient>_<0|1> (0=baseline, 1=suivi). Un patient peut avoir seulement
_0, seulement _1, ou les deux. On splitte AU NIVEAU PATIENT (les _0 et _1 d'un meme patient
vont du meme cote -> pas de fuite train/test), en stratifiant par les 3 categories pour des
tailles deterministes.

Produit DEUX vues EMBOITEES (meme partition patient) :
  reports_all_{train,test}_ids.csv        -> tous les exams (baseline + suivi)
  reports_baseline0_{train,test}_ids.csv  -> uniquement les _0 (baseline)
=> reports_baseline0_test est un SOUS-ENSEMBLE de reports_all_test.
"""
import argparse, csv, glob, os, re, random


def parse_args():
    p = argparse.ArgumentParser(description="Splits patient-level du dataset-rapports.")
    p.add_argument("--npz_dir", required=True, help="dataset_ich_reports_npz (source des stems <patient>_<0|1>)")
    p.add_argument("--splits_dir", required=True, help="dossier de sortie des CSV de split")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def write_ids(path, ids):
    # colonne 'BDMAP_ID' : c'est ce qu'attend train_ddp.py --ucsf_ids (dataset_ich_reports.py).
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["BDMAP_ID"])
        for i in sorted(ids): w.writerow([i])


def split_group(patients, frac, rng):
    pts = sorted(patients); rng.shuffle(pts)
    n_test = round(frac * len(pts))
    return set(pts[:n_test]), set(pts[n_test:])   # test, train


def main():
    a = parse_args()
    os.makedirs(a.splits_dir, exist_ok=True)
    stems = sorted(os.path.basename(f)[:-len("_gt.npz")]
                   for f in glob.glob(f"{a.npz_dir}/*_gt.npz"))
    exams = {}   # patient -> set of exams ('0'/'1')
    for s in stems:
        m = re.match(r"^(.*)_([01])$", s)
        if not m:
            print(f"  ! stem non conforme ignore: {s}"); continue
        exams.setdefault(m.group(1), set()).add(m.group(2))

    only0 = {p for p, e in exams.items() if e == {"0"}}
    only1 = {p for p, e in exams.items() if e == {"1"}}
    both  = {p for p, e in exams.items() if e == {"0", "1"}}
    print(f"patients: {len(exams)} (only_0={len(only0)}, only_1={len(only1)}, both={len(both)}) "
          f"| exams: {len(stems)}")

    rng = random.Random(a.seed)
    t0, tr0 = split_group(only0, a.test_frac, rng)
    t1, tr1 = split_group(only1, a.test_frac, rng)
    tb, trb = split_group(both,  a.test_frac, rng)
    test_pat = t0 | t1 | tb
    train_pat = tr0 | tr1 | trb
    assert not (test_pat & train_pat)

    def exams_of(pats, base_only=False):
        out = []
        for p in pats:
            for e in sorted(exams[p]):
                if base_only and e != "0": continue
                out.append(f"{p}_{e}")
        return out

    all_test  = exams_of(test_pat);  all_train  = exams_of(train_pat)
    b0_test   = exams_of(test_pat, True); b0_train = exams_of(train_pat, True)

    write_ids(f"{a.splits_dir}/reports_all_test_ids.csv", all_test)
    write_ids(f"{a.splits_dir}/reports_all_train_ids.csv", all_train)
    write_ids(f"{a.splits_dir}/reports_baseline0_test_ids.csv", b0_test)
    write_ids(f"{a.splits_dir}/reports_baseline0_train_ids.csv", b0_train)

    print(f"\nreports_all       : test={len(all_test)}  train={len(all_train)}  (patients test={len(test_pat)})")
    print(f"reports_baseline0 : test={len(b0_test)}  train={len(b0_train)}")
    # verifs
    assert set(b0_test) <= set(all_test), "baseline0_test doit etre inclus dans all_test"
    assert set(b0_train) <= set(all_train)
    pat_all_test = {re.match(r'^(.*)_[01]$', i).group(1) for i in all_test}
    pat_all_train = {re.match(r'^(.*)_[01]$', i).group(1) for i in all_train}
    assert not (pat_all_test & pat_all_train), "fuite patient train/test !"
    print("OK: baseline0_test ⊂ all_test, aucune fuite patient.")
    print(f"-> {a.splits_dir}/reports_{{all,baseline0}}_{{train,test}}_ids.csv")


if __name__ == "__main__":
    main()
