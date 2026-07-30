import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an inference sweep (checkpoint x preset) from a YAML config."
    )
    parser.add_argument(
        "config", nargs="?", default="configs/inference_sweep.yaml",
        help="sweep config YAML (default: configs/inference_sweep.yaml)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "exit non-zero if any scene job is not 'success'. OFF by default: "
            "scripts/benchmark_inference_config_sweep.py:93-97 raises RuntimeError "
            "on ANY non-zero child exit, so a default non-zero exit would abort "
            "that benchmark harness mid-run. Safe (and recommended) for the "
            "shell callers -- run_ship_inference.sh / run_site_eval.sh."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg_path = Path(args.config)
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

    # Number of checkpoint x preset groups run concurrently. 1 (default)
    # reproduces the previous strictly serial behavior with inherited stdout.
    # >1 shares the GPU between groups (each group's own outputs are unchanged
    # by contention) and redirects per-group output to logs/<group_id>.log.
    max_parallel = max(1, int(sweep.get("max_parallel", 1)))

    manifest_rows = []
    command_lines = []
    started_at_sweep = utc_now_iso()
    t0_sweep = time.time()

    group_specs = []
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

        # Forward per-checkpoint encoder_kwargs (e.g. swinv2 needs
        # img_size/strict_img_size to relax its fixed 256px patch-embed assertion).
        # The default inference model config has no encoder_kwargs key, so each
        # leaf is appended with Hydra's '+' prefix.
        for kw_key, kw_value in (checkpoint["model"].get("encoder_kwargs") or {}).items():
            overrides[f"+model.encoder_kwargs.{kw_key}"] = kw_value

        unsanitized_run_name = f"{sweep_id}__{checkpoint['name']}__{preset['name']}"
        run_name = sanitize_for_path(unsanitized_run_name)
        overrides["inference.output.run_name"] = run_name
        overrides["inference.continue_on_error"] = True

        cmd = ["python", "scripts/run_inference.py"]
        for key, value in overrides.items():
            cmd.append(f"{key}={to_cli_value(value)}")

        command = " ".join(cmd)
        command_lines.append(command)

        group_specs.append(
            {
                "group_idx": group_idx,
                "group_id": group_id,
                "run_name": run_name,
                "unsanitized_run_name": unsanitized_run_name,
                "image_list_path": image_list_path,
                "checkpoint": checkpoint,
                "preset": preset,
                "overrides": overrides,
                "cmd": cmd,
                "command": command,
            }
        )

    def execute_group(spec: dict) -> dict:
        started_at = utc_now_iso()
        t0 = time.time()
        if max_parallel > 1:
            log_path = sweep_root / "logs" / f"{spec['group_id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(spec["cmd"], stdout=log_file, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(spec["cmd"])
        status = "success" if result.returncode == 0 else "failed"
        error_message = ""
        if status == "failed":
            error_message = f"run_inference.py returned exit code {result.returncode}"
            print(f"FAILED GROUP: {spec['run_name']}")
        return {
            "status": status,
            "return_code": result.returncode,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_minutes": (time.time() - t0) / 60.0,
            "error_message": error_message,
        }

    results = {}
    if dry_run:
        for spec in group_specs:
            print(f"[{spec['group_idx']}/{len(groups)}] {spec['run_name']}")
            print(f"  Image list: {spec['image_list_path']}")
            print(spec["command"])
            now = utc_now_iso()
            results[spec["group_idx"]] = {
                "status": "dry_run", "return_code": "", "started_at": now,
                "finished_at": now, "elapsed_minutes": 0.0, "error_message": "",
            }
    elif max_parallel == 1:
        for spec in group_specs:
            print(f"[{spec['group_idx']}/{len(groups)}] {spec['run_name']}")
            print(f"  Image list: {spec['image_list_path']}")
            results[spec["group_idx"]] = execute_group(spec)
            if results[spec["group_idx"]]["status"] == "failed" and not continue_on_error:
                break
    else:
        print(f"Running up to {max_parallel} groups concurrently; per-group output -> {sweep_root / 'logs'}")
        stopping = False
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {}
            for spec in group_specs:
                print(f"[{spec['group_idx']}/{len(groups)}] queued: {spec['run_name']}")
                futures[pool.submit(execute_group, spec)] = spec
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results[spec["group_idx"]] = future.result()
                except CancelledError:
                    continue
                if results[spec["group_idx"]]["status"] == "failed" and not continue_on_error and not stopping:
                    stopping = True
                    pool.shutdown(wait=False, cancel_futures=True)

    for spec in group_specs:
        outcome = results.get(spec["group_idx"])
        if outcome is None:
            # Group never ran: an earlier group failed with continue_on_error
            # false. (The old serial runner wrote no rows for these at all.)
            outcome = {
                "status": "skipped", "return_code": "", "started_at": "",
                "finished_at": "", "elapsed_minutes": 0.0,
                "error_message": "skipped after earlier group failure",
            }
        shoreline_format = spec["overrides"].get(
            "inference.post_processing.shoreline.output_format", "gpkg"
        )

        for image_idx, image in enumerate(images, start=1):
            paths = build_scene_paths(root_dir, spec["run_name"], image, shoreline_format)
            manifest_rows.append(
                {
                    "sweep_id": sweep_id,
                    "group_id": spec["group_id"],
                    "job_id": f"{spec['group_id']}__scene_{image_idx:04d}",
                    "status": outcome["status"],
                    "return_code": outcome["return_code"],
                    "input_image": image,
                    "image_list_file": str(spec["image_list_path"]),
                    "scene_id": paths["scene_id"],
                    "checkpoint_name": spec["checkpoint"]["name"],
                    "checkpoint_path": spec["checkpoint"]["checkpoint_path"],
                    "model_arch": spec["checkpoint"]["model"]["arch"],
                    "model_encoder": spec["checkpoint"]["model"].get("encoder_name", ""),
                    "preset_name": spec["preset"]["name"],
                    "tile_size": spec["overrides"].get("inference.data.tile_size", ""),
                    "buffer_size": spec["overrides"].get("inference.data.buffer_size", ""),
                    "stride": spec["overrides"].get("inference.data.stride", ""),
                    "edge_policy": spec["overrides"].get("inference.data.edge_policy", ""),
                    "stitching_mode": spec["overrides"].get("inference.stitching.mode", ""),
                    "blend_window": spec["overrides"].get("inference.stitching.blend_window", ""),
                    "tta_enabled": spec["overrides"].get("inference.tta.enabled", False),
                    "tta_transforms": spec["overrides"].get("inference.tta.transforms", ""),
                    "run_name": spec["run_name"],
                    "unsanitized_run_name": spec["unsanitized_run_name"],
                    "scene_output_dir": paths["scene_output_dir"],
                    "probability_geotiff": paths["probability_geotiff"],
                    "shoreline_geojson": paths["shoreline_geojson"],
                    "metadata_json": paths["metadata_json"],
                    "started_at": outcome["started_at"],
                    "finished_at": outcome["finished_at"],
                    "elapsed_minutes": f"{outcome['elapsed_minutes']:.4f}",
                    "error_message": outcome["error_message"],
                    "command": spec["command"],
                }
            )

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
        # Additive provenance, for the completion check that the exit code cannot
        # give (this script exits 0 even when groups fail): the per-run scene
        # expectation and the run dirs this sweep wrote into. Compare against
        # `ls <root_dir>/<run_dir>/*/*_probability_water.tif | wc -l`, or
        # scripts/evaluation/vm/completion.py.
        "expected_scenes_per_run": len(images),
        "run_dirs": sorted({spec["run_name"] for spec in group_specs}),
        "manifest_csv": str(sweep_root / "sweep_manifest.csv"),
        "commands_txt": str(sweep_root / "commands.txt"),
    }
    write_json(sweep_root / "sweep_summary.json", summary)

    print("Sweep complete")
    print(f"Manifest: {sweep_root / 'sweep_manifest.csv'}")
    print(f"Summary: {sweep_root / 'sweep_summary.json'}")

    # Make partial failure MACHINE-VISIBLE. `continue_on_error: true` is in every
    # sweep template, so a sweep can finish short of its scene count and still
    # exit 0; without this line the only trace is a status column in the CSV.
    n_bad = counts.get("failed", 0) + counts.get("skipped", 0)
    if n_bad:
        print(f"WARNING: {n_bad} scene job(s) not successful: {counts}")
        print(
            f"  verify completeness by counting *_probability_water.tif per run dir "
            f"against {len(images)} expected scene(s): "
            f"python scripts/evaluation/vm/completion.py --expected {len(images)} "
            f"--check {root_dir}/<run_dir>"
        )
        if args.strict:
            raise SystemExit(
                f"--strict: {n_bad} scene job(s) not successful "
                f"(see {sweep_root / 'sweep_summary.json'})"
            )


if __name__ == "__main__":
    main()
