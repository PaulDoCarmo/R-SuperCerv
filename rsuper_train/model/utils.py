import numpy as np
import torch
import torch.nn as nn
import pdb

import sys
import os
sys.path.append(os.path.abspath(".."))
from training import losses_foundation as lf


def get_model(args, pretrain=False, classes=None, classes_cls=None):
    """
    R-SuperCerv n'utilise que MedFormer 3D. Le reste du zoo MedFormer/CBIM
    (UNet, VNet, UNETR, SwinUNETR, nnFormer, VTUNet, modeles 2D...) a ete retire
    pour la clarte. Si besoin d'une autre architecture, la recopier/adapter
    depuis le depot R-Super original.
    """
    if args.dimension != '3d' or args.model != 'medformer':
        raise ValueError(
            f"Seul 'medformer' en 3d est supporte ici (recu model={args.model}, "
            f"dimension={args.dimension}). Recopier l'architecture voulue depuis R-Super."
        )

    from .dim3 import MedFormer

    class_list_seg = classes

    if (classes_cls is None) and (class_list_seg is not None):
        class_list_cls = [c for c in class_list_seg
                          if (('background' in c) or ('lesion' in c) or ('pnet' in c)
                              or ('cyst' in c) or ('pdac' in c))]
    else:
        class_list_cls = classes_cls
    print('Class list seg:', class_list_seg)
    print('Class list cls:', class_list_cls)

    if classes is None:
        classes = args.classes
    else:
        classes = len(classes)

    net = MedFormer(args.in_chan, classes, args.base_chan, map_size=args.map_size, conv_block=args.conv_block,
                    conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads,
                    fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads,
                    expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type,
                    norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss,
                    classification_branch=args.classification_branch,
                    class_list_seg=class_list_seg, class_list_cls=class_list_cls, clip_branch=args.clip_loss)

    if pretrain:
        checkpoint = torch.load(args.pretrained, weights_only=False)
        pretrained_model = checkpoint['model_state_dict']
        state_dict = pretrained_model.state_dict() if hasattr(pretrained_model, 'state_dict') else pretrained_model
        net.load_state_dict(state_dict, strict=False)

    net.loss_wrapper = None  # deprecated
    net.balancer = None      # deprecated

    return net
