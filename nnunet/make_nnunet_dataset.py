#!/usr/bin/env python3
"""
make_nnunet_dataset.py -- Convertit les donnees ICH au format nnU-Net v2 (Dataset raw).

Multi-classe : 0=fond, 1=ICH, 2=IVH, 3=PHE (vos masques sont deja dans ce format).
Resolution NATIVE (pas de resampling : nnU-Net gere l'anisotropie lui-meme).

On reutilise le MEME split que MedFormer :
  - trainval (305) -> imagesTr/ + labelsTr/   (nnU-Net fera sa 5-fold CV interne dessus)
  - test (54 held-out) -> imagesTs/            (jamais vu ; sert a l'eval comparative)

Sortie : <nnUNet_raw>/Dataset<ID>_<name>/{imagesTr,labelsTr,imagesTs}/ + dataset.json
"""
import argparse
import csv
import json
import os

import numpy as np
import nibabel as nib


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--vols_dir", default="/home/pauldcrm/links/scratch/data_laurent/NIFTI/vols")
    p.add_argument("--masks_dir", default="/home/pauldcrm/links/scratch/data_laurent/NIFTI/masks")
    p.add_argument("--trainval_ids", default="/home/pauldcrm/links/scratch/R-SuperCerv/splits/trainval_ids.csv")
    p.add_argument("--test_ids", default="/home/pauldcrm/links/scratch/R-SuperCerv/splits/test_ids.csv")
    p.add_argument("--nnunet_raw", default="/home/pauldcrm/links/scratch/R-SuperCerv/nnUNet_raw")
    p.add_argument("--dataset_id", type=int, default=1)
    p.add_argument("--dataset_name", default="ICH")
    return p.parse_args()


def read_ids(path):
    ids = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            col = "BDMAP ID" if "BDMAP ID" in row else ("BDMAP_ID" if "BDMAP_ID" in row else list(row)[0])
            ids.append(row[col].strip())
    return ids


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(src), dst)


def copy_label_uint8(src, dst):
    """Copie le masque en uint8 (nnU-Net veut des labels entiers)."""
    img = nib.load(src)
    arr = np.asarray(img.dataobj).astype(np.uint8)
    hdr = img.header.copy(); hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(arr, img.affine, hdr), dst)


def main():
    a = parse_args()
    ds_dir = os.path.join(a.nnunet_raw, f"Dataset{a.dataset_id:03d}_{a.dataset_name}")
    imagesTr = os.path.join(ds_dir, "imagesTr")
    labelsTr = os.path.join(ds_dir, "labelsTr")
    imagesTs = os.path.join(ds_dir, "imagesTs")
    for d in (imagesTr, labelsTr, imagesTs):
        os.makedirs(d, exist_ok=True)

    trainval = read_ids(a.trainval_ids)
    test = read_ids(a.test_ids)
    print(f"trainval = {len(trainval)} | test = {len(test)}")

    n_tr = 0
    for cid in trainval:
        vol = os.path.join(a.vols_dir, cid + ".nii.gz")
        msk = os.path.join(a.masks_dir, cid + ".nii.gz")
        if not (os.path.exists(vol) and os.path.exists(msk)):
            print(f"  [skip trainval] manquant : {cid}"); continue
        link(vol, os.path.join(imagesTr, cid + "_0000.nii.gz"))
        copy_label_uint8(msk, os.path.join(labelsTr, cid + ".nii.gz"))
        n_tr += 1

    n_ts = 0
    for cid in test:
        vol = os.path.join(a.vols_dir, cid + ".nii.gz")
        if not os.path.exists(vol):
            print(f"  [skip test] manquant : {cid}"); continue
        link(vol, os.path.join(imagesTs, cid + "_0000.nii.gz"))
        n_ts += 1

    dataset_json = {
        "channel_names": {"0": "CT"},                  # -> nnU-Net applique CTNormalization
        "labels": {"background": 0, "ICH": 1, "IVH": 2, "PHE": 3},
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    with open(os.path.join(ds_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nDataset nnU-Net cree : {ds_dir}")
    print(f"  imagesTr/labelsTr : {n_tr} cas | imagesTs : {n_ts} cas")
    print(f"  dataset.json : labels = {dataset_json['labels']}")
    print("\nProchaines etapes (env nnU-Net) :")
    print(f"  export nnUNet_raw={a.nnunet_raw}")
    print( "  export nnUNet_preprocessed=.../nnUNet_preprocessed ; export nnUNet_results=.../nnUNet_results")
    print(f"  nnUNetv2_plan_and_preprocess -d {a.dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
