"""Zero-shot SAM automatic-mask-generation diagnostic.

No training, no fine-tuning — SAM used strictly as a pretrained, frozen mask
proposer. Checks whether SegmentAnything's automatic mask generator produces
clean, usable part-level masks (rotors/wings/booms/fuselage) on eVTOL patent
figures (mix of line drawings and rendered images) before committing to a
SAM-crops+SigLIP extraction pipeline.

Read-only: only reads images from ``sam_test.sample_dir`` and writes
visualizations/JSON under ``sam_test.output_dir``. Never touches any other
pipeline file or notebook state.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def print_gpu_status(device: str) -> None:
    """Print which device is selected and current free/total VRAM on it.

    Call this before loading the model so free VRAM can be confirmed first.
    """
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        print(f"[sam_test] device={device} (no CUDA free-memory info available)")
        return

    idx = torch.cuda.current_device() if ":" not in device else int(device.split(":")[1])
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    free_gb, total_gb = free_bytes / 2**30, total_bytes / 2**30
    name = torch.cuda.get_device_name(idx)
    print(f"[sam_test] device={device} ({name}) — {free_gb:.2f} GB free / {total_gb:.2f} GB total")


def load_mask_generator(cfg: Dict[str, Any]):
    """Load SAM's automatic mask generator from ``cfg['sam_test']``.

    Returns the ``SamAutomaticMaskGenerator`` instance. Requires the
    ``segment-anything`` package and a downloaded checkpoint matching
    ``model_type`` (not fetched by this repo — see config.yaml EDIT-ME note).
    """
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam_cfg = cfg["sam_test"]
    model_type = sam_cfg["model_type"]
    checkpoint_path = sam_cfg["checkpoint_path"]
    device = sam_cfg["device"]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint_path}. Download the "
            f"'{model_type}' checkpoint from the SegmentAnything repo and set "
            f"sam_test.checkpoint_path in config.yaml."
        )

    print_gpu_status(device)

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    print(f"[sam_test] loaded SAM ({model_type}) on {device}")

    return SamAutomaticMaskGenerator(sam)


def _sample_images(sample_dir: Path, n_samples: int, seed: int) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    candidates = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in exts)
    if not candidates:
        raise FileNotFoundError(f"No images found in {sample_dir}")
    rng = random.Random(seed)
    return rng.sample(candidates, k=min(n_samples, len(candidates)))


def _overlay_masks(image: np.ndarray, masks: List[Dict[str, Any]]):
    """Original image + semi-transparent random-colored mask overlay, via matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image)
    ax.axis("off")

    if masks:
        overlay = np.zeros((*masks[0]["segmentation"].shape, 4))
        for m in sorted(masks, key=lambda x: x["area"], reverse=True):
            color = np.concatenate([np.random.random(3), [0.45]])
            overlay[m["segmentation"]] = color
        ax.imshow(overlay)

    ax.set_title(f"{len(masks)} masks", fontsize=10)
    fig.tight_layout(pad=0.2)
    return fig


def _mask_record(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "bbox": [float(x) for x in m["bbox"]],
        "area": int(m["area"]),
        "predicted_iou": float(m["predicted_iou"]),
        "stability_score": float(m["stability_score"]),
    }


def run_sam_test(cfg: Dict[str, Any]) -> Path:
    """Run automatic mask generation over a random sample and save diagnostics.

    For each sampled image, writes:
      - ``<stem>_overlay.png``  (original + colored mask overlay)
      - ``<stem>_masks.json``   (bbox/area/predicted_iou/stability_score per mask)

    Returns the output directory.
    """
    import matplotlib.pyplot as plt

    sam_cfg = cfg["sam_test"]
    sample_dir = Path(sam_cfg["sample_dir"])
    output_dir = Path(sam_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = cfg["folder_root"] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(sam_cfg.get("n_samples", 18))
    seed = int(sam_cfg.get("seed", 42))
    images = _sample_images(sample_dir, n_samples, seed)
    print(f"[sam_test] sampled {len(images)} images from {sample_dir}")

    mask_generator = load_mask_generator(cfg)

    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))
        masks = mask_generator.generate(image)
        print(f"[sam_test] {img_path.name}: {len(masks)} masks")

        fig = _overlay_masks(image, masks)
        fig.savefig(output_dir / f"{img_path.stem}_overlay.png")
        plt.close(fig)

        records = [_mask_record(m) for m in masks]
        with open(output_dir / f"{img_path.stem}_masks.json", "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    print(f"[sam_test] wrote overlays + mask JSON for {len(images)} images to {output_dir}")
    return output_dir
