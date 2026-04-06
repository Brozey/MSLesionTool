"""
nnUNetTrainer_WandB_Focal — Focal Loss variant for class imbalance.

Replaces standard Cross-Entropy with Focal Loss (Lin et al., 2017).
Focal Loss applies a modulating factor (1 - p_t)^gamma to the CE loss,
downweighting easy-to-classify voxels (mostly background) and focusing
on hard examples (small lesions, boundaries, ambiguous regions).

This is the loss function strategy used by MadSeg (MSLesSeg-2024 winner),
who used Focal Loss variants to address the extreme class imbalance in
MS lesion segmentation.

Inherits all features from nnUNetTrainer_WandB (augmentation, W&B logging,
early stopping, VRAM probing, gradient checkpointing, etc.).

Usage:
    python -u -m nnunetv2.run.run_training DatasetXXX 3d_fullres FOLD \
        -tr nnUNetTrainer_WandB_Focal -p nnUNetResEncUNetLPlans
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WandB import nnUNetTrainer_WandB


class FocalLoss(nn.Module):
    """Focal Loss for multi-class segmentation (Lin et al., RetinaNet 2017).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    With gamma=2 (default), a well-classified voxel with p_t=0.9 gets
    weight (1-0.9)^2 = 0.01 (100x downweighted), while a misclassified
    voxel with p_t=0.1 gets weight (1-0.1)^2 = 0.81 (nearly full weight).
    """

    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight          # optional per-class weight tensor
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: (B, C, *spatial) logits
        # target: (B, 1, *spatial) — nnU-Net convention
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        ce_loss = F.cross_entropy(
            input, target, weight=self.weight,
            ignore_index=self.ignore_index, reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class DC_and_Focal_loss(nn.Module):
    """Compound loss: Dice + Focal Loss.

    Same structure as nnU-Net's DC_and_CE_loss but replaces CE with Focal.
    """

    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_ce=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        if ignore_label is not None:
            focal_kwargs['ignore_index'] = ignore_label

        self.ce = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss


class nnUNetTrainer_WandB_Focal(nnUNetTrainer_WandB):
    """nnUNetTrainer_WandB with Dice + Focal Loss (gamma=2).

    Focal Loss downweights easy-to-classify voxels by factor (1-p_t)^gamma,
    focusing training on hard examples: small lesion boundaries, ambiguous
    voxels, and false negatives. Combined with Dice for overlap optimization.

    This mirrors the loss strategy of MadSeg (MSLesSeg-2024 challenge winner),
    who used Focal Loss variants to address class imbalance.
    """

    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_WARMUP = 100

    def _build_loss(self):
        loss = DC_and_Focal_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            focal_kwargs={'gamma': 2.0},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
