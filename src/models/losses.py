import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(logits, targets.float())


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        p = torch.sigmoid(logits)

        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1,
                             torch.full_like(p, self.alpha),
                             torch.full_like(p, 1 - self.alpha))

        modulation = (1 - pt) ** self.gamma
        focal_loss = alpha_t * modulation * ce_loss

        return focal_loss.mean()


class AuxiliaryLoss(nn.Module):
    def __init__(self, main_loss_fn, aux_weight=0.3):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.aux_weight = aux_weight

    def forward(self, main_logit, targets, aux_logits_dict=None):
        main_loss = self.main_loss_fn(main_logit, targets)

        if aux_logits_dict is None or len(aux_logits_dict) == 0:
            return main_loss

        aux_losses = [
            F.binary_cross_entropy_with_logits(aux_logit, targets.float())
            for aux_logit in aux_logits_dict.values()
        ]
        aux_loss = torch.stack(aux_losses).mean()

        return main_loss + self.aux_weight * aux_loss
