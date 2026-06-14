import segmentation_models_pytorch as smp
from typing import Optional

class SegmentationModelFactory:
    @staticmethod
    def build(arch: str, encoder_name: str, in_channels: int, classes: int,  encoder_weights: Optional[str] = "imagenet", **encoder_kwargs):
        a = arch.lower()
        if a == "unet":
            return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a in ("unetplusplus", "unet++"):
            return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a in ("deeplabv3plus", "deeplabv3+"):
            return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a in ('DPT','dpt'):
            return smp.DPT(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a == "segformer":
            return smp.Segformer(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a == "upernet":
            return smp.UPerNet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, **encoder_kwargs)
        elif a == "upernet-highres":
            return smp.UPerNet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, output_stride=8, **encoder_kwargs)
        else:
            raise ValueError(f"Unsupported arch '{arch}'.")
