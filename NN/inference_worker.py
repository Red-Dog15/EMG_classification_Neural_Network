"""
NN Inference subprocess worker.

Runs under the Scripts/.venv Python environment (PyTorch, pandas available).
Launched by simulation/run_nn.py via subprocess.Popen — do NOT run directly
unless you want to test the protocol manually.

Protocol (newline-delimited JSON over stdin/stdout):
  Parent → worker  stdin  : {"window": [[f, f, ...] * 8] * window_size}
                          | {"command": "exit"}
  Worker → parent  stdout : {"movement_name": str, "movement_confidence": float,
                              "movement_pred": int}
  Worker → parent  stdout : {"status": "ready"}   (once, after model is loaded)

stderr is forwarded to the terminal — safe to print debug info there.
"""

import argparse
import json
import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
_NN_DIR = os.path.dirname(os.path.abspath(__file__))          # Scripts/NN/
_SCRIPTS_DIR = os.path.dirname(_NN_DIR)                        # Scripts/
_DATA_DIR = os.path.join(_SCRIPTS_DIR, "DATA")

for _p in (_SCRIPTS_DIR, _NN_DIR, _DATA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from predict import load_trained_model, predict_from_tensor


def _err(msg: str) -> None:
    """Write to stderr (visible in parent terminal, never pollutes stdout JSON)."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="EMG inference worker (subprocess)")
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()

    _err(f"[inference_worker] Loading model: {args.model}")
    if not os.path.isfile(args.model):
        _err(f"[inference_worker] ERROR: model file not found: {args.model}")
        sys.exit(1)

    # Redirect stdout → stderr during model load so print() calls inside
    # load_trained_model() don't pollute the JSON-lines stdout channel.
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        model, _ = load_trained_model(args.model)
    device = next(model.parameters()).device
    _err(f"[inference_worker] Ready on device: {device}")

    # Signal to parent that the model is loaded and we are ready.
    print(json.dumps({"status": "ready"}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _err(f"[inference_worker] JSON decode error: {exc}")
            continue

        if msg.get("command") == "exit":
            _err("[inference_worker] Received exit command.")
            break

        window_data = msg.get("window")
        if window_data is None:
            _err("[inference_worker] Received message with no 'window' key — skipping.")
            continue

        window_tensor = torch.tensor(window_data, dtype=torch.float32)
        results = predict_from_tensor(
            model, window_tensor, window_size=args.window_size, device=device
        )

        response = {
            "movement_name": results["movement_name"],
            "movement_pred": int(results["movement_pred"]),
            "movement_confidence": float(results["movement_confidence"]),
            "severity_name": results["severity_name"],
            "severity_pred": int(results["severity_pred"]),
            "severity_confidence": float(results["severity_confidence"]),
        }
        print(json.dumps(response), flush=True)

    _err("[inference_worker] Exiting.")


if __name__ == "__main__":
    main()
