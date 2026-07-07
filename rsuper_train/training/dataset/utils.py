import numpy as np


def get_dataset(args, mode, **kwargs):
    """
    R-SuperCerv : deux datasets (3D uniquement) :
      - 'ich'     -> stage 1 (segmentation, masques seuls)
      - 'ich_ufo' -> stage 2 (supervision par rapports : Volume/Ball loss)
    Les datasets du framework CBIM (acdc, lits, bcv, kits, amos...) ont ete
    retires pour la clarte (recopiables depuis R-Super si besoin).
    """
    if args.dimension != '3d':
        raise ValueError("R-SuperCerv ne supporte que la 3D.")

    if args.dataset == 'ich':
        from .dim3.dataset_ich import ICHDataset
        return ICHDataset(args, mode=mode, seed=args.split_seed,
                          all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                          load_augmented=args.load_augmented, save_destination=args.save_destination,
                          save_augmented=args.save_augmented)

    elif args.dataset == 'ich_ufo':
        from .dim3.dataset_ich_reports import ICHReportsDataset
        if args.pancreas_only:
            tumor_classes = ['pancreas']
        elif args.kidney_only:
            tumor_classes = ['kidney']
        elif hasattr(args, 'tumor_classes'):
            tumor_classes = args.tumor_classes
        else:
            tumor_classes = None

        ds_kwargs = dict(mode=mode, seed=args.split_seed, all_train=args.all_train,
                         crop_on_tumor=args.crop_on_tumor, load_augmented=args.load_augmented,
                         save_destination=args.save_destination, save_augmented=args.save_augmented,
                         Atlas_only=args.Atlas_only, UFO_only=args.UFO_only)
        if tumor_classes is not None:
            ds_kwargs['tumor_classes'] = tumor_classes
        return ICHReportsDataset(args, **ds_kwargs)

    raise ValueError(f"Dataset non supporte: {args.dataset} (attendu: ich, ich_ufo).")
