import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_for_path(value: str) -> str:
    value = str(value).strip().replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def scene_id_from_path(path: str) -> str:
    return sanitize_for_path(Path(path).stem)


def to_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def resolve_images(sweep: dict) -> list[str]:
    images = list(sweep.get("input_images", []))
    if images:
        return images
    if sweep.get("input_dir"):
        return sorted(str(p) for p in Path(sweep["input_dir"]).glob(sweep.get("input_glob", "*.tif")))
    return []


def build_scene_paths(root_dir: str, run_name: str, image: str, shoreline_format: str = "gpkg") -> dict:
    scene_id = scene_id_from_path(image)
    scene_dir = Path(root_dir) / run_name / scene_id
    shoreline_ext = "geojson" if str(shoreline_format).lower() == "geojson" else "gpkg"
    return {
        "scene_id": scene_id,
        "scene_output_dir": str(scene_dir),
        "probability_geotiff": str(scene_dir / f"{scene_id}_probability_water.tif"),
        "shoreline_geojson": str(scene_dir / f"{scene_id}_shoreline.{shoreline_ext}"),
        "metadata_json": str(scene_dir / f"{scene_id}_metadata.json"),
    }


def main():
    cfg_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/inference_sweep.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    sweep = cfg["sweep"]

    sweep_name = sanitize_for_path(sweep["name"])
    sweep_id = f"{sweep_name}__{timestamp_id()}"
    dry_run = bool(sweep.get("dry_run", False))
    continue_on_error = bool(sweep.get("continue_on_error", True))

    sweep_root = Path(sweep.get("output_root", "outputs/inference/sweeps")) / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, sweep_root / "sweep_config.yaml")

    images = resolve_images(sweep)
    if not images:
        raise ValueError("No input images found for sweep.")

    common = sweep.get("common_overrides", {})
    root_dir = str(common.get("inference.output.root_dir", "outputs/inference/runs"))

    groups = []
    for checkpoint in sweep["checkpoints"]:
        for preset in sweep["presets"]:
            groups.append((checkpoint, preset))

    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep folder: {sweep_root}")
    print(f"Images: {len(images)}")
    print(f"Process groups checkpoint x preset: {len(groups)}")
    print(f"Expected checkpoint loads: {len(groups)}")
    print(f"Scene-level jobs: {len(images) * len(groups)}")

    manifest_rows = []
    command_lines = []
    started_at_sweep = utc_now_iso()
    t0_sweep = time.time()

    for group_idx, (checkpoint, preset) in enumerate(groups, start=1):
        group_id = f"group_{group_idx:04d}__{sanitize_for_path(checkpoint['name'])}__{sanitize_for_path(preset['name'])}"
        image_list_path = sweep_root / "image_lists" / f"{group_id}.txt"
        image_list_path.parent.mkdir(parents=True, exist_ok=True)
        image_list_path.write_text("\n".join(images) + "\n", encoding="utf-8")

        overrides = {}
        overrides.update(common)
        overrides.update(preset["overrides"])
        overrides["inference.data.input_list_file"] = str(image_list_path)
        overrides["inference.checkpoint_path"] = checkpoint["checkpoint_path"]
        overrides["model.arch"] = checkpoint["model"]["arch"]
        if checkpoint["model"].get("encoder_name"):
            overrides["model.encoder_name"] = checkpoint["model"]["encoder_name"]

        unsanitized_run_name = f"{sweep_id}__{checkpoint['name']}__{preset['name']}"
        run_name = sanitize_for_path(unsanitized_run_name)
        overrides["inference.output.run_name"] = run_name
        overrides["inference.continue_on_error"] = True

        cmd = ["python", "scripts/run_inference.py"]
        for key, value in overrides.items():
            cmd.append(f"{key}={to_cli_value(value)}")

        command = " ".join(cmd)
        command_lines.append(command)

        print(f"[{group_idx}/{len(groups)}] {run_name}")
        print(f"  Image list: {image_list_path}")
        if dry_run:
            print(command)

        started_at = utc_now_iso()
        t0 = time.time()
        return_code = ""
        error_message = ""

        if dry_run:
            status = "dry_run"
            finished_at = utc_now_iso()
            elapsed_minutes = 0.0
        else:
            result = subprocess.run(cmd)
            return_code = result.returncode
            finished_at = utc_now_iso()
            elapsed_minutes = (time.time() - t0) / 60.0
            status = "success" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                error_message = f"run_inference.py returned exit code {result.returncode}"
                print(f"FAILED GROUP: {run_name}")

        shoreline_format = overrides.get(
            "inference.post_processing.shoreline.output_format", "gpkg"
        )

        for image_idx, image in enumerate(images, start=1):
            paths = build_scene_paths(root_dir, run_name, image, shoreline_format)
            manifest_rows.append(
                {
                    "sweep_id": sweep_id,
                    "group_id": group_id,
                    "job_id": f"{group_id}__scene_{image_idx:04d}",
                    "status": status,
                    "return_code": return_code,
                    "input_image": image,
                    "image_list_file": str(image_list_path),
                    "scene_id": paths["scene_id"],
                    "checkpoint_name": checkpoint["name"],
                    "checkpoint_path": checkpoint["checkpoint_path"],
                    "model_arch": checkpoint["model"]["arch"],
                    "model_encoder": checkpoint["model"].get("encoder_name", ""),
                    "preset_name": preset["name"],
                    "tile_size": overrides.get("inference.data.tile_size", ""),
                    "buffer_size": overrides.get("inference.data.buffer_size", ""),
                    "stride": overrides.get("inference.data.stride", ""),
                    "edge_policy": overrides.get("inference.data.edge_policy", ""),
                    "stitching_mode": overrides.get("inference.stitching.mode", ""),
                    "blend_window": overrides.get("inference.stitching.blend_window", ""),
                    "tta_enabled": overrides.get("inference.tta.enabled", False),
                    "tta_transforms": overrides.get("inference.tta.transforms", ""),
                    "run_name": run_name,
                    "unsanitized_run_name": unsanitized_run_name,
                    "scene_output_dir": paths["scene_output_dir"],
                    "probability_geotiff": paths["probability_geotiff"],
                    "shoreline_geojson": paths["shoreline_geojson"],
                    "metadata_json": paths["metadata_json"],
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_minutes": f"{elapsed_minutes:.4f}",
                    "error_message": error_message,
                    "command": command,
                }
            )

        if status == "failed" and not continue_on_error:
            break

    (sweep_root / "commands.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    fieldnames = [
        "sweep_id", "group_id", "job_id", "status", "return_code",
        "input_image", "image_list_file", "scene_id", "checkpoint_name",
        "checkpoint_path", "model_arch", "model_encoder", "preset_name",
        "tile_size", "buffer_size", "stride", "edge_policy", "stitching_mode",
        "blend_window", "tta_enabled", "tta_transforms", "run_name",
        "unsanitized_run_name", "scene_output_dir", "probability_geotiff",
        "shoreline_geojson", "metadata_json", "started_at", "finished_at",
        "elapsed_minutes", "error_message", "command",
    ]
    write_csv(sweep_root / "sweep_manifest.csv", manifest_rows, fieldnames)

    counts = {}
    for row in manifest_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    summary = {
        "sweep_id": sweep_id,
        "sweep_name": sweep_name,
        "dry_run": dry_run,
        "started_at": started_at_sweep,
        "finished_at": utc_now_iso(),
        "elapsed_minutes": (time.time() - t0_sweep) / 60.0,
        "num_images": len(images),
        "num_process_groups": len(groups),
        "num_scene_jobs": len(manifest_rows),
        "checkpoint_loads_expected": len(groups),
        "status_counts": counts,
        "manifest_csv": str(sweep_root / "sweep_manifest.csv"),
        "commands_txt": str(sweep_root / "commands.txt"),
    }
    write_json(sweep_root / "sweep_summary.json", summary)

    print("Sweep complete")
    print(f"Manifest: {sweep_root / 'sweep_manifest.csv'}")
    print(f"Summary: {sweep_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
