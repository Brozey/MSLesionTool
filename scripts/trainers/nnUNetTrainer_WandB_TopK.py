"""
nnUNetTrainer_WandB_TopK — TopK loss variant for hard-example mining.

Replaces standard Cross-Entropy with TopK CE (top 10% hardest voxels).
This focuses gradient updates on the most difficult voxels: small lesion
boundaries, ambiguous regions, and false negatives — exactly the voxels
that standard CE averages away due to the massive class imbalance in
MS lesion segmentation (~1-5% lesion vs ~95-99% background).

Inherits all features from nnUNetTrainer_WandB (augmentation, W&B logging,
early stopping, VRAM probing, gradient checkpointing, etc.).

Gradient accumulation: 2 micro-batches per optimizer step, doubling the
effective batch size without increasing VRAM usage.

Usage:
    python -u -m nnunetv2.run.run_training DatasetXXX 3d_fullres FOLD \
        -tr nnUNetTrainer_WandB_TopK -p nnUNetResEncUNetLPlans
"""

import numpy as np
import torch
from torch.cuda.amp import autocast

from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WandB import nnUNetTrainer_WandB
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainer_WandB_TopK(nnUNetTrainer_WandB):
    """nnUNetTrainer_WandB with Dice + TopK CE loss and gradient accumulation.

    TopK(k=10) selects only the hardest 10% of voxels by CE loss value,
    discarding the 90% of easy, correctly-classified background voxels.
    Combined with Dice loss for volumetric overlap optimization.

    Gradient accumulation with ACCUMULATION_STEPS micro-batches per optimizer
    step. Effective batch = VRAM-probed batch * ACCUMULATION_STEPS.
    """

    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_WARMUP = 100
    ACCUMULATION_STEPS = 2

    def _build_loss(self):
        loss = DC_and_topk_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            ce_kwargs={'k': 10},     # top 10% hardest voxels
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
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

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"Gradient accumulation: {self.ACCUMULATION_STEPS} micro-batches per step "
            f"(effective batch = {self.batch_size} x {self.ACCUMULATION_STEPS} = "
            f"{self.batch_size * self.ACCUMULATION_STEPS})"
        )

    def train_step(self, batch: dict) -> dict:
        """Training step with gradient accumulation.

        Runs ACCUMULATION_STEPS micro-batches, accumulating gradients,
        then does a single optimizer step. The first micro-batch is the
        one passed in; additional micro-batches are fetched from the dataloader.
        """
        accum = self.ACCUMULATION_STEPS
        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        for micro in range(accum):
            if micro == 0:
                b = batch
            else:
                b = next(self.dataloader_train)

            data = b['data'].to(self.device, non_blocking=True)
            target = b['target']
            if isinstance(target, list):
                target = [i.to(self.device, non_blocking=True) for i in target]
            else:
                target = target.to(self.device, non_blocking=True)

            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)
                l = self.loss(output, target) / accum

            if self.grad_scaler is not None:
                self.grad_scaler.scale(l).backward()
            else:
                l.backward()

            total_loss += l.detach().cpu().numpy()

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss * accum}  # report un-averaged loss for consistency
