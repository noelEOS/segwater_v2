import torch
import torchmetrics

class SegmentationMetrics:
    """Wrapper around torchmetrics for multi-class semantic segmentation ignoring a specific index.

    With cm_only=True only the ConfusionMatrix is updated per batch and
    mIoU/f1_macro are derived from it at compute() time — the confusion matrix
    already contains everything they need, so per-batch cost drops to a single
    metric update. Matches the full mode's macro semantics, including NaN when
    a class has zero union/support.
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 255, device: str = "cpu", cm_only: bool = False):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device
        self.cm_only = cm_only

        if not cm_only:
            self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)
            self.f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)
        self.cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if not self.cm_only:
            self.iou.update(preds, target)
            self.f1_macro.update(preds, target)
        self.cm.update(preds, target)

    def compute(self) -> dict:
        cm_val = self.cm.compute()

        if self.cm_only:
            cm_f = cm_val.to(torch.float64)
            diag = cm_f.diagonal()
            pred_totals = cm_f.sum(dim=0)
            target_totals = cm_f.sum(dim=1)
            union = pred_totals + target_totals - diag
            # Macro-average over classes actually present (torchmetrics
            # ignores absent classes); NaN only when no class is present.
            miou = (diag[union > 0] / union[union > 0]).mean()
            f1_denom = pred_totals + target_totals
            f1 = (2 * diag[f1_denom > 0] / f1_denom[f1_denom > 0]).mean()
        else:
            miou = self.iou.compute()
            f1 = self.f1_macro.compute()

        result = {
            "mIoU": miou.item(),
            "f1_macro": f1.item(),
        }

        C = cm_val.shape[0]
        if C == 2:
            tn, fp, fn, tp = cm_val[0, 0], cm_val[0, 1], cm_val[1, 0], cm_val[1, 1]
            result.update({
                "TN": tn.item(),
                "FP": fp.item(),
                "FN": fn.item(),
                "TP": tp.item()
            })
        else:
            for i in range(C):
                for j in range(C):
                    result[f"CM_{i}_{j}"] = cm_val[i, j].item()

        return result

    def reset(self):
        if not self.cm_only:
            self.iou.reset()
            self.f1_macro.reset()
        self.cm.reset()
