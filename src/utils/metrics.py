import torch
import torchmetrics

class SegmentationMetrics:
    """Wrapper around torchmetrics for multi-class semantic segmentation ignoring a specific index."""
    
    def __init__(self, num_classes: int = 2, ignore_index: int = 255, device: str = "cpu"):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device
        
        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)
        self.f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)
        self.cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.iou.update(preds, target)
        self.f1_macro.update(preds, target)
        self.cm.update(preds, target)

    def compute(self) -> dict:
        miou = self.iou.compute()
        f1 = self.f1_macro.compute()
        cm_val = self.cm.compute()
        
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
        self.iou.reset()
        self.f1_macro.reset()
        self.cm.reset()
