"""Download all LTX-2.3 models and LoRAs from Hugging Face.

Target hardware (verified via nvidia-smi): NVIDIA GeForce RTX 5090, 32 GB VRAM,
consumer Blackwell. The 22B base checkpoints (~46 GB bf16 each) exceed 32 GB
VRAM and therefore require `--quantization fp8-cast` at inference time; this
script downloads them regardless since the runnability check passed under fp8.

Run:
    python download_models.py --dry-run        # preview + total size, no transfers
    python download_models.py                  # download (asks for confirmation)
    python download_models.py --yes            # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


@dataclass
class Item:
    category: str
    subdir: str
    repo_id: str
    filename: str | None
    size: int | None
    label: str


GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_SHARD_SIZES = {
    "model-00001-of-00005.safetensors": 4_979_902_192,
    "model-00002-of-00005.safetensors": 4_931_296_592,
    "model-00003-of-00005.safetensors": 4_931_296_656,
    "model-00004-of-00005.safetensors": 4_931_296_656,
    "model-00005-of-00005.safetensors": 4_601_000_928,
    "tokenizer.json": 33_384_570,
    "tokenizer.model": 4_689_074,
}
GEMMA_TOTAL = sum(GEMMA_SHARD_SIZES.values())


def _build_manifest() -> list[Item]:
    items: list[Item] = []
    main_repo = "Lightricks/LTX-2.3"

    items.append(Item("Main checkpoint", "LTX-2.3", main_repo, "ltx-2.3-22b-dev.safetensors", 46_149_344_974, "Dev (full-quality; two-stage pipeline)"))
    items.append(Item("Main checkpoint", "LTX-2.3", main_repo, "ltx-2.3-22b-distilled-1.1.safetensors", 46_149_345_334, "Distilled 1.1 (fast 8-step; DistilledPipeline)"))

    items.append(Item("Spatial upscaler", "LTX-2.3", main_repo, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", 995_743_560, "Spatial upscaler x2 v1.1 (required for two-stage)"))
    items.append(Item("Spatial upscaler", "LTX-2.3", main_repo, "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors", 1_090_125_794, "Spatial upscaler x1.5 v1.0"))
    items.append(Item("Temporal upscaler", "LTX-2.3", main_repo, "ltx-2.3-temporal-upscaler-x2-1.0.safetensors", 261_944_000, "Temporal upscaler x2 v1.0"))

    items.append(Item("Distilled LoRA", "LTX-2.3", main_repo, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors", 7_605_507_256, "Distilled LoRA 384 v1.1 (required for two-stage)"))

    items.append(Item("Text encoder", "gemma-3-12b-it-qat-q4_0-unquantized", GEMMA_REPO, None, GEMMA_TOTAL, "Gemma 3 12B (whole repo)"))

    items.append(Item("IC-LoRA", "loras/IC-LoRA-Union-Control", "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", 654_465_352, "Union Control"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-Motion-Track-Control", "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control", "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors", 327_309_314, "Motion Track Control"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-Detailer", "Lightricks/LTX-2-19b-IC-LoRA-Detailer", "ltx-2-19b-ic-lora-detailer.safetensors", 2_617_401_920, "Detailer"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-Pose-Control", "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", "ltx-2-19b-ic-lora-pose-control.safetensors", 654_465_256, "Pose Control"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-HDR", "Lightricks/LTX-2.3-22b-IC-LoRA-HDR", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors", 327_309_312, "HDR v0.9"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-HDR", "Lightricks/LTX-2.3-22b-IC-LoRA-HDR", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors", 12_583_096, "HDR scene embeddings"))
    items.append(Item("IC-LoRA", "loras/IC-LoRA-LipDub", "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub", "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors", 2_466_665_072, "LipDub v0.9"))

    camera = [
        ("Dolly-In", "ltx-2-19b-lora-camera-control-dolly-in.safetensors", 327_309_208),
        ("Dolly-Left", "ltx-2-19b-lora-camera-control-dolly-left.safetensors", 327_309_208),
        ("Dolly-Out", "ltx-2-19b-lora-camera-control-dolly-out.safetensors", 327_309_208),
        ("Dolly-Right", "ltx-2-19b-lora-camera-control-dolly-right.safetensors", 327_309_208),
        ("Jib-Down", "ltx-2-19b-lora-camera-control-jib-down.safetensors", 2_214_978_664),
        ("Jib-Up", "ltx-2-19b-lora-camera-control-jib-up.safetensors", 2_214_978_664),
        ("Static", "ltx-2-19b-lora-camera-control-static.safetensors", 2_214_978_664),
    ]
    for name, fname, size in camera:
        items.append(Item("Camera LoRA", f"loras/Camera-Control-{name}", f"Lightricks/LTX-2-19b-LoRA-Camera-Control-{name}", fname, size, f"Camera {name}"))

    return items


def _human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if isinstance(n, float) else f"{n} {unit}"
        n = n / 1024.0
    return f"{n:.2f} TB"


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(str(path))
    return usage.free


def _print_manifest(items: list[Item], models_dir: Path) -> None:
    print(f"\nModels directory: {models_dir}\n")
    by_cat: dict[str, list[Item]] = {}
    for it in items:
        by_cat.setdefault(it.category, []).append(it)
    total = 0
    for cat, group in by_cat.items():
        cat_total = sum(i.size or 0 for i in group)
        total += cat_total
        print(f"[{cat}]  ({_human(cat_total)})")
        for it in group:
            tgt = models_dir / it.subdir / (it.filename or "<whole repo>")
            print(f"    {it.label:<40} {it.repo_id}")
            print(f"      -> {tgt}   {_human(it.size)}")
        print()
    print(f"Total to download: {_human(total)}")
    print(f"Free on {models_dir.anchor}: {_human(_free_bytes(models_dir.parent))}\n")


def _download_item(it: Item, models_dir: Path) -> None:
    target_dir = models_dir / it.subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    if it.filename is None:
        print(f"\n>>> [whole repo] {it.repo_id}  ->  {target_dir}  ({_human(it.size)})")
        snapshot_download(
            repo_id=it.repo_id,
            local_dir=str(target_dir),
            max_workers=4,
            etag_timeout=60,
        )
        return
    print(f"\n>>> {it.filename}  ({_human(it.size)})")
    print(f"    {it.repo_id} -> {target_dir}")
    hf_hub_download(
        repo_id=it.repo_id,
        filename=it.filename,
        local_dir=str(target_dir),
        etag_timeout=60,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download LTX-2.3 models from Hugging Face.")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"), help="Destination directory for downloaded models.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded and exit.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    items = _build_manifest()
    _print_manifest(items, models_dir)

    total = sum(i.size or 0 for i in items)
    free = _free_bytes(models_dir.parent)
    if total > free:
        print(f"ERROR: not enough free space. Need {_human(total)}, have {_human(free)} on {models_dir.anchor}.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run only — no files downloaded.")
        return 0

    if not args.yes:
        answer = input(f"Download {_human(total)} now? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    failures: list[tuple[Item, str]] = []
    for idx, it in enumerate(items, 1):
        print(f"\n=== [{idx}/{len(items)}] {it.category}: {it.label} ===")
        try:
            _download_item(it, models_dir)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            failures.append((it, str(exc)))

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min.  Success: {len(items) - len(failures)}/{len(items)}.")
    if failures:
        print("\nFailures:")
        for it, msg in failures:
            print(f"  - {it.label} ({it.repo_id}/{it.filename}): {msg}")
        return 3

    print("\nLayout:")
    for it in items:
        tgt = models_dir / it.subdir / (it.filename or "<repo>")
        print(f"  {tgt}")
    print("\nReminder: load 22B checkpoints with --quantization fp8-cast (32 GB VRAM).")
    print("Reminder: install a CUDA build of torch + xFormers before inference (current venv is torch+cpu).")
    return 0


if __name__ == "__main__":
    sys.exit(main())