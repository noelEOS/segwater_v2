from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


def flatten_overrides(d: dict, prefix=""):
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_overrides(v, key))
        else:
            items[key] = v
    return items


cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/inference_sweep.yaml"
cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

sweep = cfg["sweep"]

images = list(sweep.get("input_images", []))
if not images and sweep.get("input_dir"):
    images = sorted(
        str(p)
        for p in Path(sweep["input_dir"]).glob(sweep.get("input_glob", "*.tif"))
    )

common = sweep.get("common_overrides", {})

jobs = []
for image in images:
    for checkpoint in sweep["checkpoints"]:
        for preset in sweep["presets"]:
            jobs.append((image, checkpoint, preset))

print(f"Generated {len(jobs)} sweep jobs")

for idx, (image, checkpoint, preset) in enumerate(jobs, start=1):
    cmd = ["python", "scripts/run_inference.py"]

    overrides = {}
    overrides.update(common)
    overrides.update(preset["overrides"])

    overrides["inference.data.input_image"] = image
    overrides["inference.checkpoint_path"] = checkpoint["checkpoint_path"]
    overrides["model.arch"] = checkpoint["model"]["arch"]

    if checkpoint["model"].get("encoder_name"):
        overrides["model.encoder_name"] = checkpoint["model"]["encoder_name"]

    run_name = f"{sweep['name']}__{checkpoint['name']}__{preset['name']}"
    overrides["inference.output.run_name"] = run_name

    print(f"[{idx}/{len(jobs)}] {run_name}")

    for key, value in overrides.items():
        cmd.append(f"{key}={value}")

    if sweep.get("dry_run", False):
        print(" ".join(cmd))
        continue

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"FAILED: {run_name}")
        if not sweep.get("continue_on_error", True):
            raise SystemExit(result.returncode)

print("Sweep complete")
