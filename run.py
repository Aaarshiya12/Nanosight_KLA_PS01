"""SEMICON KLA PS01 final submission entry point.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy grayscale input, restores it with the bundled trained model,
and writes one .npy output with the same filename. Outputs are float32,
finite, in [0, 1], and have 2x the input spatial resolution.
"""

from pathlib import Path
import sys

import numpy as np
import torch

# Allow the existing project modules to remain under src/.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import build_model
from noise_estimator import estimate_noise_map


MODEL_PATH = ROOT / "models" / "kla_best_model.pt"
MODEL_SIZE = "small"


def load_input(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)

    # Accept (H, W) or (H, W, 1); reject color/multi-channel arrays.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(
            f"{path.name}: expected grayscale shape (H,W) or (H,W,1), got {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"{path.name}: input contains NaN or Inf values")

    # Match the training pipeline: KLA arrays may be normalized to ~[0,1]
    # or represented on a 0..255 scale. Speckle can legitimately create
    # values outside the nominal range, so do not clip the input.
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > 4.0:
        arr = arr / 255.0

    if arr.size == 0:
        raise ValueError(f"{path.name}: input array is empty")

    h, w = arr.shape
    if h < 16 or w < 16 or h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"{path.name}: input shape {arr.shape} must have dimensions divisible by 16"
        )

    return arr


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        raise SystemExit(2)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model weights not found: {MODEL_PATH}")

    model = build_model(MODEL_SIZE).to(device).eval()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded trained weights from: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Found {len(files)} input .npy files.")

    with torch.inference_mode():
        for path in files:
            arr = load_input(path)
            h, w = arr.shape

            noise = estimate_noise_map(arr, patch_size=8).astype(np.float32)

            image_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            noise_tensor = torch.from_numpy(noise).unsqueeze(0).unsqueeze(0).to(device)

            restored, _ = model(image_tensor, noise_tensor)
            out = restored.squeeze().detach().cpu().numpy().astype(np.float32)

            # Required output contract: 2x spatial resolution, grayscale,
            # finite values in [0,1].
            expected_shape = (2 * h, 2 * w)
            if out.shape != expected_shape:
                raise RuntimeError(
                    f"{path.name}: model returned {out.shape}, expected {expected_shape}"
                )

            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0).astype(np.float32)

            output_path = output_dir / path.name
            np.save(output_path, out)

            # Final hard validation of the organizer's output contract.
            check = np.load(output_path)
            if check.shape not in (expected_shape, expected_shape + (1,)):
                raise RuntimeError(f"{path.name}: invalid saved output shape {check.shape}")
            if not np.isfinite(check).all():
                raise RuntimeError(f"{path.name}: saved output contains NaN or Inf")
            if float(check.min()) < 0.0 or float(check.max()) > 1.0:
                raise RuntimeError(f"{path.name}: saved output is outside [0,1]")

            print(
                f"{path.name} -> {output_path.name} | "
                f"input {arr.shape} -> output {check.shape} | "
                f"range [{check.min():.4f}, {check.max():.4f}]"
            )

    print(f"Completed: {len(files)} files written to {output_dir}")


if __name__ == "__main__":
    main()
