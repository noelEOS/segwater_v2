import csv
import json
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
    import re

    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def scene_id_from_path(path: str) -> str:
    return sanitize_for_path(Path(path).stem)


def flatten_overrides(d: dict, prefix=""):
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_overrides(v, key))
        else:
            items[key] = v
    return items


def to_cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def get_override(overrides: dict, key: str, default=""):
    return overrides.get(key, default)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def build_output_paths(root_dir: str, run_name: str, image: str):
    scene_id = scene_id_from_path(image)
    scene_dir = Path(root_dir) / run_name / scene_id
    return {
        "scene_id": scene_id,
        "scene_output_dir": str(scene_dir),
        "probability_geotiff": str(scene_dir / f"{scene_id}_probability_water.tif"),
        "shoreline_geojson": str(scene_dir / f"{scene_id}_shoreline.geojson"),
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

    archived_config_path = sweep_root / "sweep_config.yaml"
    shutil.copy2(cfg_path, archived_config_path)

    manifest_path = sweep_root / "sweep_manifest.csv"
    summary_path = sweep_root / "sweep_summary.json"
    commands_path = sweep_root / "commands.txt"

    images = list(sweep.get("input_images", []))
    if not images and sweep.get("input_dir"):
        images = sorted(
            str(p)
            for p in Path(sweep["input_dir"]).glob(sweep.get("input_glob", "*.tif"))
        )

    common = sweep.get("common_overrides", {})
    root_dir = str(common.get("inference.output.root_dir", "outputs/inference/runs"))

    jobs = []
    for image in images:
        for checkpoint in sweep["checkpoints"]:
            for preset in sweep["presets"]:
                jobs.append((image, checkpoint, preset))

    print(f"Sweep ID: {sweep_id}")
    print(f"Sweep folder: {sweep_root}")
    print(f"Generated {len(jobs)} sweep jobs")

    manifest_rows = []
    command_lines = []
    sweep_started_at = utc_now_iso()
    sweep_start_time = time.time()

    for idx, (image, checkpoint, preset) in enumerate(jobs, start=1):
        job_id = f"job_{idx:04d}"
        cmd = ["python", "scripts/run_inference.py"]

        overrides = {}
        overrides.update(common)
        overrides.update(preset["overrides"])

        overrides["inference.data.input_image"] = image
        overrides["inference.checkpoint_path"] = checkpoint["checkpoint_path"]
        overrides["model.arch"] = checkpoint["model"]["arch"]

        if checkpoint["model"].get("encoder_name"):
            overrides["model.encoder_name"] = checkpoint["model"]["encoder_name"]

        run_name = f"{sweep_id}__{checkpoint['name']}__{preset['name']}"
        overrides["inference.output.run_name"] = run_name

        output_paths = build_output_paths(root_dir, run_name, image)

        for key, value in overrides.items():
            cmd.append(f"{key}={to_cli_value(value)}")

        command_str = " ".join(cmd)
        command_lines.append(command_str)

        print(f"[{idx}/{len(jobs)}] {run_name}")
        print(command_str if dry_run else "")

        started_at = utc_now_iso()
        job_start = time.time()
        status = "dry_run" if dry_run else "running"
        return_code = ""
        error_message = ""

        if dry_run:
            finished_at = utc_now_iso()
            elapsed_minutes = 0.0
        else:
            result = subprocess.run(cmd)
            return_code = result.returncode
            finished_at = utc_now_iso()
            elapsed_minutes = (time.time() - job_start) / 60.0

            if result.returncode == 0:
                status = "success"
            else:
                status = "failed"
                error_message = f"run_inference.py returned non-zero exit code: {result.returncode}"
                print(f"FAILED: {run_name}")
                if not continue_on_error:
                    manifest_rows.append(
                        build_manifest_row(
                            sweep_id,
                            job_id,
                            status,
                            image,
                            checkpoint,
                            preset,
                            overrides,
                            run_name,
                            output_paths,
                            started_at,
                            finished_at,
                            elapsed_minutes,
                            return_code,
                            error_message,
                            command_str,
                        )
                    )
                    break

        manifest_rows.append(
            build_manifest_row(
                sweep_id,
                job_id,
                status,
                image,
                checkpoint,
                preset,
                overrides,
                run_name,
                output_paths,
                started_at,
                finished_at,
                elapsed_minutes,
                return_code,
                error_message,
                command_str,
            )
        )

        if status == "failed" and not continue_on_error:
            break

    commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    fieldnames = [
        "sweep_id",
        "job_id",
        "status",
        "return_code",
        "input_image",
        "scene_id",
        "checkpoint_name",
        "checkpoint_path",
        "model_arch",
        "model_encoder",
        "preset_name",
        "tile_size",
        "buffer_size",
        "stride",
        "edge_policy",
        "stitching_mode",
        "blend_window",
        "tta_enabled",
        "tta_transforms",
        "run_name",
        "scene_output_dir",
        "probability_geotiff",
        "shoreline_geojson",
        "metadata_json",
        "started_at",
        "finished_at",
        "elapsed_minutes",
        "error_message",
        "command",
    ]
    write_csv(manifest_path, manifest_rows, fieldnames)

    counts = {}
    for row in manifest_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    sweep_finished_at = utc_now_iso()
    summary = {
        "sweep_id": sweep_id,
        "sweep_name": sweep_name,
        "dry_run": dry_run,
        "started_at": sweep_started_at,
        "finished_at": sweep_finished_at,
        "elapsed_minutes": (time.time() - sweep_start_time) / 60.0,
        "num_jobs_requested": len(jobs),
        "num_jobs_recorded": len(manifest_rows),
        "status_counts": counts,
        "archived_config": str(archived_config_path),
        "manifest_csv": str(manifest_path),
        "commands_txt": str(commands_path),
    }
    write_json(summary_path, summary)

    print("Sweep complete")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Commands: {commands_path}")


def build_manifest_row(
    sweep_id,
    job_id,
    status,
    image,
    checkpoint,
    preset,
    overrides,
    run_name,
    output_paths,
    started_at,
    finished_at,
    elapsed_minutes,
    return_code,
    error_message,
    command_str,
):
    return {
        "sweep_id": sweep_id,
        "job_id": job_id,
        "status": status,
        "return_code": return_code,
        "input_image": image,
        "scene_id": output_paths["scene_id"],
        "checkpoint_name": checkpoint["name"],
        "checkpoint_path": checkpoint["checkpoint_path"],
        "model_arch": checkpoint["model"]["arch"],
        "model_encoder": checkpoint["model"].get("encoder_name", ""),
        "preset_name": preset["name"],
        "tile_size": get_override(overrides, "inference.data.tile_size"),
        "buffer_size": get_override(overrides, "inference.data.buffer_size"),
        "stride": get_override(overrides, "inference.data.stride"),
        "edge_policy": get_override(overrides, "inference.data.edge_policy"),
        "stitching_mode": get_override(overrides, "inference.stitching.mode"),
        "blend_window": get_override(overrides, "inference.stitching.blend_window"),
        "tta_enabled": get_override(overrides, "inference.tta.enabled", False),
        "tta_transforms": get_override(overrides, "inference.tta.transforms", ""),
        "run_name": run_name,
        "scene_output_dir": output_paths["scene_output_dir"],
        "probability_geotiff": output_paths["probability_geotiff"],
        "shoreline_geojson": output_paths["shoreline_geojson"],
        "metadata_json": output_paths["metadata_json"],
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_minutes": f"{elapsed_minutes:.4f}",
        "error_message": error_message,
        "command": command_str,
    }


if __name__ == "__main__":
    main()
