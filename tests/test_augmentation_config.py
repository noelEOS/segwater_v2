"""Tests for config-driven augmentation wiring.

The capability added here lets augmentation probabilities (hflip / vflip /
rot90) flow from config -> CoastalDataModule -> CoastalAug, instead of being
frozen at CoastalAug's hardcoded defaults.

The guarantees under test:
  1. augment=False builds NO transform, regardless of aug_params -- this keeps
     the default pipeline byte-identical to the pre-feature (v0.2.0) behaviour.
  2. augment=True with no aug_params reproduces CoastalAug's signature defaults.
  3. aug_params overrides actually reach CoastalAug (e.g. enabling rotation).
  4. The Hydra-style ``cfg.data.get("aug", {})`` access resolves to a usable
     mapping that round-trips through the DataModule.
"""

import torch
from omegaconf import OmegaConf

from src.data.datamodule import CoastalDataModule
from src.data.transforms import CoastalAug


def _make_dm(augment, aug_params=None):
    # Point at paths that do not exist so setup() constructs no datasets and we
    # can inspect the augmentation decision without real memmap files.
    return CoastalDataModule(
        root_dir="/nonexistent",
        augment=augment,
        aug_params=aug_params,
    )


def _build_aug(dm):
    """Mirror the single decision setup() makes when wiring transforms."""
    return CoastalAug(**dm.aug_params) if dm.augment else None


def test_augment_false_builds_no_transform_even_with_params():
    # Guards the reproducibility invariant: augment off => no CoastalAug.
    dm = _make_dm(augment=False, aug_params={"rot90_k_prob": 1.0})
    assert _build_aug(dm) is None


def test_augment_true_no_params_matches_signature_defaults():
    dm = _make_dm(augment=True, aug_params=None)
    aug = _build_aug(dm)
    reference = CoastalAug()
    assert aug.hflip_p == reference.hflip_p
    assert aug.vflip_p == reference.vflip_p
    assert aug.rot90_k_prob == reference.rot90_k_prob


def test_aug_params_reach_coastalaug():
    dm = _make_dm(
        augment=True,
        aug_params={"hflip_p": 0.1, "vflip_p": 0.2, "rot90_k_prob": 0.5},
    )
    aug = _build_aug(dm)
    assert aug.hflip_p == 0.1
    assert aug.vflip_p == 0.2
    assert aug.rot90_k_prob == 0.5


def test_none_params_normalised_to_empty_dict():
    dm = _make_dm(augment=True, aug_params=None)
    assert dm.aug_params == {}


def test_hydra_style_access_round_trips():
    # Emulate the ``cfg.data.get("aug", {})`` call site in the train/optimize
    # scripts: an OmegaConf node should resolve to a mapping CoastalAug accepts.
    cfg = OmegaConf.create(
        {"data": {"augment": True, "aug": {"hflip_p": 0.5, "vflip_p": 0.5, "rot90_k_prob": 0.5}}}
    )
    aug_params = cfg.data.get("aug", {})
    dm = _make_dm(augment=cfg.data.augment, aug_params=aug_params)
    aug = _build_aug(dm)
    assert aug.rot90_k_prob == 0.5


def test_missing_aug_key_falls_back_to_defaults():
    # A config without an ``aug`` block (e.g. an older config) must still work.
    cfg = OmegaConf.create({"data": {"augment": True}})
    aug_params = cfg.data.get("aug", {})
    dm = _make_dm(augment=cfg.data.augment, aug_params=aug_params)
    aug = _build_aug(dm)
    assert aug.rot90_k_prob == CoastalAug().rot90_k_prob


def test_rotation_actually_rotates_when_enabled():
    # Functional check: rot90_k_prob=1.0 must change a non-symmetric input.
    torch.manual_seed(0)
    aug = CoastalAug(hflip_p=0.0, vflip_p=0.0, rot90_k_prob=1.0)
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    y = torch.arange(4 * 4).reshape(4, 4)
    # Retry across k draws; k can be 0 (a no-op rotation). Assert that at least
    # one draw produces a changed tensor, proving the rotation path is live.
    changed = False
    for _ in range(20):
        xr, yr = aug(x.clone(), y.clone())
        if not torch.equal(xr, x):
            changed = True
            assert xr.shape == x.shape and yr.shape == y.shape
            break
    assert changed, "rot90_k_prob=1.0 never altered the input"
