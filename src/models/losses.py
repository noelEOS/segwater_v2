import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CoastalCompositeLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = 255, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.dice_loss = smp.losses.LovaszLoss(
            mode="multiclass", ignore_index=ignore_index
        )
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = self.ce_loss(logits, targets)
        
        if self.dice_weight > 0.0:
            dl = self.dice_loss(logits, targets)
            total_loss = self.ce_weight * ce + self.dice_weight * dl
            return {"loss": total_loss, "loss_ce": ce, "loss_dice": dl}
            
        return {"loss": ce, "loss_ce": ce, "loss_dice": torch.tensor(0.0, device=logits.device)}
