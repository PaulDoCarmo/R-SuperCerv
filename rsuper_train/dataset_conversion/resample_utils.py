"""
resample_utils.py -- helpers SimpleITK pour le resampling 3D.

Extrait (verbatim) des fonctions generiques de R-Super
(rsuper_train/dataset_conversion/utils.py). Aucune hypothese specifique a
l'abdomen : ce sont des utilitaires de reechantillonnage / reorientation.
On ne garde que les 3 fonctions utilisees par le pipeline ICH.
"""

import numpy as np
import SimpleITK as sitk


def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    """Reechantillonne une image vers l'espacement `space` (mm), en gardant
    origine et orientation. `interp` = sitkBSpline pour une image, sitkNearestNeighbor
    pour un label ou pour un axe anisotrope qu'on ne veut pas lisser."""
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0] * sp1[0] * 1.0 / space[0])),
           int(round(sz1[1] * sp1[1] * 1.0 / space[1])),
           int(round(sz1[2] * sp1[2] * 1.0 / space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    return sitk.Resample(imImage, imRefImage, identity1, interp)


def ResampleLabelToRef(imLabel, imRef, interp=sitk.sitkNearestNeighbor):
    """Reechantillonne un label sur exactement la grille d'une image de reference."""
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())

    return sitk.Resample(imLabel, imRefImage, identity1, interp)


def reorient_image(image, desired_orientation='RAI'):
    """Reoriente l'image vers une orientation canonique (RAI par defaut)."""
    current = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())
    if current != desired_orientation:
        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation(desired_orientation)
        image = f.Execute(image)
    return image
