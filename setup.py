from setuptools import setup, find_packages

setup(
    name="coastal-water-segmentation",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "optuna",
        "wandb",
        "hydra-core",
        "torchmetrics",
        "numpy",
    ],
    description="Coastal Water Segmentation refactored modular pure PyTorch codebase.",
)
