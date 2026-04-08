import torch

class CoastalAug:
    """Torch-only geometric augs: random flips and 90 degree rotations."""

    def __init__(self, hflip_p=0.5, vflip_p=0.0, rot90_k_prob=0.5):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot90_k_prob = rot90_k_prob

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (2,H,W) float, y: (H,W) long
        if torch.rand(()) < self.hflip_p:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        if torch.rand(()) < self.vflip_p:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
        if torch.rand(()) < self.rot90_k_prob:
            k = int(torch.randint(0, 4, (1,)))
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(0, 1))
        return x, y
