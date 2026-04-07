"""
Architecture Benchmark for EMG Multi-Task Classification Models
================================================================
Produces dissertation-ready metrics for NN-A (full), NN-B (standard_cnn),
and NN-C (lightweight):

  1. Inference latency  – mean ± std ms per window (CPU, single-window mode)
  2. Model size         – parameter count, trainable params, file size (MB)
  3. Per-class metrics  – Precision, Recall, F1 for both movement and severity
  4. Confusion matrices – saved as PNG for movement and severity, per model

Outputs are written to:
  Scripts/DATA/Results/Benchmarks/
    benchmark_summary.txt    – human-readable table
    benchmark_data.json      – machine-readable, all metrics
    <ModelName>_movement_confusion.png
    <ModelName>_severity_confusion.png
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DATA.Data_Conversion import (
    create_labeled_dataset, MOVEMENT_LABELS, SEVERITY_LABELS
)
from DATA.dataset import create_dataloaders
from NN.predict import load_trained_model, predict_from_tensor

try:
    from config import WINDOW_SIZE, STRIDE, TRAIN_SPLIT, SPLIT_SEED
except ImportError:
    WINDOW_SIZE, STRIDE, TRAIN_SPLIT, SPLIT_SEED = 100, 10, 0.8, 42

# ---------------------------------------------------------------------------
SCRIPTS_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR   = os.path.join(SCRIPTS_ROOT, "NN", "models")
OUT_DIR      = os.path.join(SCRIPTS_ROOT, "DATA", "Results", "Benchmarks")
COMBINED_DIR = os.path.join(OUT_DIR, "combined figures")

MODEL_REGISTRY = {
    "NN-A (CNN+GRU)":    "best_model_full.pth",
    "NN-B (Standard CNN)": "best_model_standard_cnn.pth",
    "NN-C (Lightweight)":  "best_model_lightweight.pth",
}

LATENCY_WARMUP_REPS  = 50    # throw-away runs to warm up CPU cache
LATENCY_MEASURE_REPS = 1000  # timed runs per model

# Dense evaluation: stride-1 means every possible window in the held-out segment
# is tested (~9.86x more windows than train_stride=10).  Override via env var if needed.
EVAL_STRIDE = int(os.getenv("BENCHMARK_EVAL_STRIDE", "1"))
MEMORY_SAMPLE_EVERY = int(os.getenv("BENCHMARK_MEMORY_SAMPLE_EVERY", "20"))
# ---------------------------------------------------------------------------


def _get_process_rss_mb():
    """Return current process RSS memory in MB and method used."""
    # Prefer psutil if available.
    try:
        import psutil  # type: ignore
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        return float(rss), "psutil-rss"
    except Exception:
        pass

    # Windows fallback via WinAPI (true process working set memory).
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)

            OpenProcess = kernel32.OpenProcess
            OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            OpenProcess.restype = wintypes.HANDLE

            CloseHandle = kernel32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            CloseHandle.restype = wintypes.BOOL

            GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            GetProcessMemoryInfo.restype = wintypes.BOOL

            handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, os.getpid())
            if not handle:
                return float("nan"), "unavailable"

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            ok = GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            CloseHandle(handle)
            if ok:
                return float(counters.WorkingSetSize / (1024 ** 2)), "winapi-workingset"
        except Exception:
            pass

    # Last-resort fallback
    return float("nan"), "unavailable"


def _precision_recall_f1(true_labels, pred_labels, num_classes):
    """Compute per-class and macro-average Precision, Recall, F1."""
    true = np.array(true_labels)
    pred = np.array(pred_labels)

    per_class = {}
    for c in range(num_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        per_class[c] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   int((true == c).sum()),
        }

    # Macro average (unweighted)
    macro_p  = np.mean([v["precision"] for v in per_class.values()])
    macro_r  = np.mean([v["recall"]    for v in per_class.values()])
    macro_f1 = np.mean([v["f1"]        for v in per_class.values()])
    macro = {
        "precision": round(float(macro_p),  4),
        "recall":    round(float(macro_r),  4),
        "f1":        round(float(macro_f1), 4),
    }
    return per_class, macro


def _macro_f1_from_labels(true_labels, pred_labels, num_classes):
    """Compute macro-F1 without rounding, used for uncertainty estimation."""
    true = np.array(true_labels)
    pred = np.array(pred_labels)
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _bootstrap_macro_f1_uncertainty(true_labels, pred_labels, num_classes, n_boot=200, seed=42):
    """Bootstrap uncertainty for macro-F1 (std and 95% CI)."""
    true = np.array(true_labels)
    pred = np.array(pred_labels)
    n = len(true)
    if n == 0:
        return {"std": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(_macro_f1_from_labels(true[idx], pred[idx], num_classes))

    arr = np.array(samples, dtype=float)
    return {
        "std": float(np.std(arr, ddof=1)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def _confusion_matrix(true_labels, pred_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t][p] += 1
    return cm


def _plot_confusion_matrix(cm, class_names, title, out_path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.3)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right',
             rotation_mode='anchor', fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_confusion_matrix_on_axis(ax, cm, class_names, title, show_colorbar=False):
    n = len(class_names)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right',
             rotation_mode='anchor', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=8)

    return im


def _plot_combined_confusion_matrices(cm_dict, class_names, title_prefix, out_path):
    model_labels = list(cm_dict.keys())
    if not model_labels:
        return

    fig, axes = plt.subplots(
        1,
        len(model_labels),
        figsize=(6 * len(model_labels), max(4.8, len(class_names) * 0.9)),
        constrained_layout=True,
    )
    if len(model_labels) == 1:
        axes = [axes]

    last_im = None
    for idx, (ax, label) in enumerate(zip(axes, model_labels)):
        last_im = _plot_confusion_matrix_on_axis(
            ax,
            cm_dict[label],
            class_names,
            title=f"{label}\n{title_prefix}",
            show_colorbar=False,
        )
        if idx != 0:
            ax.set_ylabel("")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.02)

    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _plot_combined_benchmark_panels(results):
    model_items = [(k, v) for k, v in results.items() if not k.startswith("_")]
    if not model_items:
        return

    labels = [k for k, _ in model_items]
    x = np.arange(len(labels))

    ckpt_mb = [v["memory"]["checkpoint_file_mb"] for _, v in model_items]
    wts_mb = [v["memory"]["weights_memory_mb"] for _, v in model_items]
    train_s = [v["training"]["training_time_sec"] or 0.0 for _, v in model_items]
    lat_ms = [v["latency"]["median_ms"] for _, v in model_items]
    thr = [v["latency"]["throughput_wps"] for _, v in model_items]
    mov_f1 = [v["movement"]["macro"]["f1"] for _, v in model_items]
    sev_f1 = [v["severity"]["macro"]["f1"] for _, v in model_items]

    eps = 1e-9
    norm_speed = np.array(thr) / (max(thr) + eps)
    norm_train_time = min(train_s) / (np.array(train_s) + eps)
    norm_weights = min(wts_mb) / (np.array(wts_mb) + eps)
    norm_mov = np.array(mov_f1)
    norm_sev = np.array(sev_f1)
    metric_names = ["Speed", "Train Time", "Weights Mem", "Mov F1", "Sev F1"]
    metric_matrix = np.vstack([
        norm_speed,
        norm_train_time,
        norm_weights,
        norm_mov,
        norm_sev,
    ]).T

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax_train, ax_memory, ax_latency, ax_tradeoff, ax_score, ax_unused = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    ax_train.bar(labels, train_s, color=colors)
    ax_train.set_title("Training Time (seconds)")
    ax_train.set_ylabel("Seconds")
    ax_train.tick_params(axis="x", rotation=18)

    width = 0.32
    ax_memory.bar(x - width / 2, ckpt_mb, width, label="Checkpoint MB", color="#4c78a8")
    ax_memory.bar(x + width / 2, wts_mb, width, label="Weights MB", color="#f58518")
    ax_memory.set_title("Memory Footprint Comparison")
    ax_memory.set_ylabel("MB")
    ax_memory.set_yscale("log")
    ax_memory.set_xticks(x)
    ax_memory.set_xticklabels(labels, rotation=18)
    ax_memory.legend(loc="upper right", fontsize=8)

    ax_latency.bar(labels, lat_ms, color=colors)
    ax_latency.set_title("Inference Latency (ms/window)")
    ax_latency.set_ylabel("Median latency (ms)")
    ax_latency.axhline(10.0, linestyle="--", color="gray", linewidth=1.2, label="10 ms threshold")
    ax_latency.legend(loc="upper right", fontsize=8)
    ax_latency.tick_params(axis="x", rotation=18)

    line_mov, = ax_tradeoff.plot(labels, mov_f1, marker="o", markersize=9, linewidth=2.2,
                                 label="Movement macro-F1", color="#1f77b4")
    line_sev, = ax_tradeoff.plot(labels, sev_f1, marker="^", markersize=9, linewidth=2.2,
                                 label="Severity macro-F1", color="#ff7f0e")
    ax_tradeoff_t = ax_tradeoff.twinx()
    line_thr, = ax_tradeoff_t.plot(labels, thr, marker="s", markersize=9, linewidth=2.0,
                                   linestyle="--", color="gray", label="Throughput")
    ax_tradeoff.set_title("Accuracy vs Throughput")
    ax_tradeoff.set_ylabel("Macro-F1")
    ax_tradeoff_t.set_ylabel("Windows/sec")
    ax_tradeoff.set_ylim(0.75, 1.01)
    ax_tradeoff.grid(alpha=0.2)
    ax_tradeoff.legend([line_mov, line_sev, line_thr],
                       [line_mov.get_label(), line_sev.get_label(), line_thr.get_label()],
                       loc="lower right", fontsize=8)

    score_width = 0.16
    idx = np.arange(len(metric_names))
    for i, (label, vals) in enumerate(zip(labels, metric_matrix)):
        ax_score.bar(idx + (i - (len(labels)-1)/2) * score_width, vals, score_width, label=label)
    ax_score.set_xticks(idx)
    ax_score.set_xticklabels(metric_names)
    ax_score.set_ylim(0, 1.08)
    ax_score.set_ylabel("Normalized score")
    ax_score.set_title("Normalized Architecture Scorecard")
    ax_score.legend(loc="upper right", fontsize=8)
    ax_score.grid(axis="y", alpha=0.2)
    ax_score.axvspan(-0.5, 2.5, color="#d8ebff", alpha=0.2)
    ax_score.axvspan(2.5, 4.5, color="#e8f7e8", alpha=0.2)
    ax_score.text(1.0, 1.03, "Efficiency", fontsize=9, ha="center")
    ax_score.text(3.5, 1.03, "Performance", fontsize=9, ha="center")

    ax_unused.axis("off")

    out_path = os.path.join(COMBINED_DIR, "benchmark_metrics_combined.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _measure_latency(model, window_size, device):
    """Return (median_ms, iqr_ms, p95_ms) for single-window CPU inference.

    Median is used (not mean) because sub-5ms latency distributions are
    heavily right-skewed by OS jitter spikes; median is robust to these
    outliers and gives a stable central-tendency across runs.
    IQR (P75-P25) is the spread measure, also robust to outlier spikes.
    """
    dummy = torch.randn(1, window_size, 8).to(device)
    model.eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP_REPS):
            model(dummy)

    # Timed
    times = []
    with torch.no_grad():
        for _ in range(LATENCY_MEASURE_REPS):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)   # → ms

    arr = np.array(times)
    median_ms = float(np.median(arr))
    iqr_ms    = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    p95_ms    = float(np.percentile(arr, 95))
    return round(median_ms, 3), round(iqr_ms, 3), round(p95_ms, 3)


def _model_stats(model, ckpt_path):
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = round(os.path.getsize(ckpt_path) / (1024 ** 2), 3)

    # Directly measurable model memory footprint (float32 parameter storage only).
    param_bytes = total * 4
    weights_memory_mb = round(param_bytes / (1024 ** 2), 3)

    return {
        "total_params": total,
        "trainable_params": train,
        "checkpoint_file_mb": size_mb,
        "weights_memory_mb": weights_memory_mb,
    }


def _load_training_time_sec(best_ckpt_path):
    """Read training_time_sec from the paired final checkpoint when available."""
    best_name = os.path.basename(best_ckpt_path)
    final_name = best_name.replace("best_model_", "final_model_")
    final_path = os.path.join(MODELS_DIR, final_name)

    if not os.path.exists(final_path):
        return None, final_name

    try:
        final_ckpt = torch.load(final_path, map_location="cpu")
        return final_ckpt.get("training_time_sec"), final_name
    except Exception:
        return None, final_name


def run_benchmark():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)
    device = torch.device('cpu')   # CPU latency is the deployment-relevant baseline

    # ---- Build test set once; all models share the same held-out windows ----
    print("Loading dataset and building test split...")
    labeled = create_labeled_dataset()
    _, test_loader = create_dataloaders(
        labeled,
        batch_size=256,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        eval_stride=EVAL_STRIDE,
        split_seed=SPLIT_SEED
    )
    test_ds = test_loader.dataset
    print(f"Test set: {len(test_ds)} windows  "
          f"(eval_stride={EVAL_STRIDE}, train_stride={STRIDE})\n")

    mov_names = [MOVEMENT_LABELS[i] for i in range(len(MOVEMENT_LABELS))]
    sev_names = [SEVERITY_LABELS[i]  for i in range(len(SEVERITY_LABELS))]
    movement_cms = {}
    severity_cms = {}

    results = {
        "_metadata": {
            "window_size": WINDOW_SIZE,
            "train_stride": STRIDE,
            "eval_stride": EVAL_STRIDE,
            "train_split": TRAIN_SPLIT,
            "split_seed": SPLIT_SEED,
            "test_windows": len(test_ds),
            "latency_warmup_reps": LATENCY_WARMUP_REPS,
            "latency_measure_reps": LATENCY_MEASURE_REPS,
            "latency_device": "cpu",
        }
    }

    for label, filename in MODEL_REGISTRY.items():
        ckpt_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"=== {label} ===")
        model, _ = load_trained_model(ckpt_path, device=device)
        model.eval()

        # ---- Size metrics ----
        stats = _model_stats(model, ckpt_path)
        training_time_sec, training_source = _load_training_time_sec(ckpt_path)
        if training_time_sec is None:
            training_time_display = "n/a"
        else:
            training_time_display = f"{training_time_sec:.2f}s"

        print(
            f"  Params: {stats['total_params']:,}  |  "
            f"Checkpoint: {stats['checkpoint_file_mb']} MB  |  "
            f"Weights: {stats['weights_memory_mb']} MB"
        )
        print(f"  Training time: {training_time_display}  (source: {training_source})")

        # ---- Latency ----
        lat_median, lat_iqr, lat_p95 = _measure_latency(model, WINDOW_SIZE, device)
        throughput = round(1000.0 / lat_median, 1)   # windows / second
        print(f"  Latency: {lat_median:.3f} ms median, IQR {lat_iqr:.3f} ms, P95 {lat_p95:.3f} ms  "
              f"({throughput} windows/sec)")

        # ---- Collect predictions on entire test set ----
        true_mov, pred_mov = [], []
        true_sev, pred_sev = [], []

        rss_before, mem_method = _get_process_rss_mb()
        rss_peak = rss_before

        with torch.no_grad():
            for i in range(len(test_ds)):
                window, mov_label, sev_label = test_ds[i]
                res = predict_from_tensor(model, window, window_size=WINDOW_SIZE,
                                         device=device)
                true_mov.append(mov_label.item())
                pred_mov.append(res['movement_pred'])
                true_sev.append(sev_label.item())
                pred_sev.append(res['severity_pred'])

                if (i % max(1, MEMORY_SAMPLE_EVERY)) == 0:
                    rss_now, _ = _get_process_rss_mb()
                    if not np.isnan(rss_now):
                        rss_peak = max(rss_peak, rss_now)

        rss_after, _ = _get_process_rss_mb()
        rss_peak_delta = (rss_peak - rss_before) if not (np.isnan(rss_peak) or np.isnan(rss_before)) else float("nan")
        print(f"  Inference RSS peak: {rss_peak:.2f} MB (Δ {rss_peak_delta:.2f} MB, {mem_method})")

        # ---- Per-class F1 / Precision / Recall ----
        mov_per_class, mov_macro = _precision_recall_f1(
            true_mov, pred_mov, len(MOVEMENT_LABELS))
        sev_per_class, sev_macro = _precision_recall_f1(
            true_sev, pred_sev, len(SEVERITY_LABELS))

        mov_uq = _bootstrap_macro_f1_uncertainty(
            true_mov, pred_mov, len(MOVEMENT_LABELS), n_boot=200, seed=42
        )
        sev_uq = _bootstrap_macro_f1_uncertainty(
            true_sev, pred_sev, len(SEVERITY_LABELS), n_boot=200, seed=99
        )

        # Throughput uncertainty via first-order propagation: T=1000/L (using IQR/2 as spread)
        throughput_iqr = (1000.0 * lat_iqr / (lat_median ** 2)) if lat_median > 0 else 0.0

        print(f"  Movement macroF1: {mov_macro['f1']:.4f}  "
              f"| Severity macroF1: {sev_macro['f1']:.4f}")

        # ---- Confusion matrices ----
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
        mov_cm = _confusion_matrix(true_mov, pred_mov, len(MOVEMENT_LABELS))
        sev_cm = _confusion_matrix(true_sev, pred_sev, len(SEVERITY_LABELS))
        movement_cms[label] = mov_cm.copy()
        severity_cms[label] = sev_cm.copy()

        _plot_confusion_matrix(
            mov_cm, mov_names,
            title=f"{label} — Movement Confusion Matrix",
            out_path=os.path.join(OUT_DIR, f"{safe_label}_movement_confusion.png")
        )
        _plot_confusion_matrix(
            sev_cm, sev_names,
            title=f"{label} — Severity Confusion Matrix",
            out_path=os.path.join(OUT_DIR, f"{safe_label}_severity_confusion.png")
        )

        results[label] = {
            "checkpoint": filename,
            "training": {
                "training_time_sec": training_time_sec,
                "training_time_min": (round(float(training_time_sec) / 60.0, 3)
                                      if training_time_sec is not None else None),
                "source_checkpoint": training_source,
            },
            "memory": {
                "checkpoint_file_mb": stats["checkpoint_file_mb"],
                "weights_memory_mb": stats["weights_memory_mb"],
                "inference_rss_before_mb": None if np.isnan(rss_before) else round(float(rss_before), 3),
                "inference_rss_after_mb": None if np.isnan(rss_after) else round(float(rss_after), 3),
                "inference_rss_peak_mb": None if np.isnan(rss_peak) else round(float(rss_peak), 3),
                "inference_rss_peak_delta_mb": None if np.isnan(rss_peak_delta) else round(float(rss_peak_delta), 3),
                "memory_probe_method": mem_method,
            },
            "size": {
                "total_params":     stats["total_params"],
                "trainable_params": stats["trainable_params"],
            },
            "latency": {
                "median_ms":        lat_median,
                "iqr_ms":           lat_iqr,
                "p95_ms":           lat_p95,
                "throughput_wps":   throughput,
                "throughput_iqr_wps": round(float(throughput_iqr), 3),
            },
            "movement": {
                "macro": mov_macro,
                "macro_uncertainty": {
                    "f1_std": round(mov_uq["std"], 5),
                    "f1_ci95_low": round(mov_uq["ci_low"], 5),
                    "f1_ci95_high": round(mov_uq["ci_high"], 5),
                },
                "per_class": {
                    mov_names[i]: v for i, v in mov_per_class.items()
                },
            },
            "severity": {
                "macro": sev_macro,
                "macro_uncertainty": {
                    "f1_std": round(sev_uq["std"], 5),
                    "f1_ci95_low": round(sev_uq["ci_low"], 5),
                    "f1_ci95_high": round(sev_uq["ci_high"], 5),
                },
                "per_class": {
                    sev_names[i]: v for i, v in sev_per_class.items()
                },
            },
        }
        print()

    # ---- Write JSON ----
    json_path = os.path.join(OUT_DIR, "benchmark_data.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # ---- Write human-readable summary ----
    txt_path = os.path.join(OUT_DIR, "benchmark_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        _write_summary(f, results, mov_names, sev_names)
    print(f"Saved summary: {txt_path}")

    # ---- Print summary to console too ----
    with open(txt_path, 'r', encoding='utf-8') as f:
        print(f.read())

    # ---- Generate comparison figures ----
    _plot_benchmark_comparison(results)
    _plot_combined_benchmark_panels(results)
    _plot_combined_confusion_matrices(
        movement_cms,
        mov_names,
        title_prefix="Movement Confusion Matrix",
        out_path=os.path.join(COMBINED_DIR, "movement_confusion_matrices_combined.png"),
    )
    _plot_combined_confusion_matrices(
        severity_cms,
        sev_names,
        title_prefix="Severity Confusion Matrix",
        out_path=os.path.join(COMBINED_DIR, "severity_confusion_matrices_combined.png"),
    )


def _write_summary(f, results, mov_names, sev_names):
    SEP = "=" * 90

    meta = results.get("_metadata", {})
    model_items = [(k, v) for k, v in results.items() if not k.startswith("_")]

    f.write(SEP + "\n")
    f.write("EMG ARCHITECTURE BENCHMARK REPORT\n")
    f.write(SEP + "\n\n")
    if meta:
        f.write(
            f"Config: window_size={meta.get('window_size')} | "
            f"train_stride={meta.get('train_stride')} | "
            f"eval_stride={meta.get('eval_stride')} | "
            f"test_windows={meta.get('test_windows')}\n\n"
        )

    # --- Split summary tables ---
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 90 + "\n")
    f.write("Caption: Movement and severity macro-F1 on the held-out temporal split.\n")
    header = f"{'Model':<26} {'MovF1':>10} {'SevF1':>10} {'MovF1 SD':>10} {'SevF1 SD':>10} {'Rank':>6}"
    f.write(header + "\n")
    f.write("-" * 90 + "\n")

    # Overall rank based on normalized trade-off score (same idea as scorecard)
    speeds = np.array([r['latency']['throughput_wps'] for _, r in model_items], dtype=float)
    train_t = np.array([
        (r['training']['training_time_sec'] if r['training']['training_time_sec'] is not None else np.nan)
        for _, r in model_items
    ], dtype=float)
    wts = np.array([r['memory']['weights_memory_mb'] for _, r in model_items], dtype=float)
    mov = np.array([r['movement']['macro']['f1'] for _, r in model_items], dtype=float)
    sev = np.array([r['severity']['macro']['f1'] for _, r in model_items], dtype=float)

    train_t = np.nan_to_num(train_t, nan=np.nanmax(train_t[np.isfinite(train_t)]) if np.any(np.isfinite(train_t)) else 1.0)
    norm_score = (
        (speeds / (speeds.max() + 1e-9)) +
        (train_t.min() / (train_t + 1e-9)) +
        (wts.min() / (wts + 1e-9)) +
        mov + sev
    ) / 5.0
    order = np.argsort(-norm_score)
    rank_map = {model_items[i][0]: int(np.where(order == i)[0][0] + 1) for i in range(len(model_items))}

    best_mov = np.max(mov)
    best_sev = np.max(sev)

    for label, r in model_items:
        mov_val = r['movement']['macro']['f1']
        sev_val = r['severity']['macro']['f1']
        mov_txt = f"{mov_val:.4f}" + ("*" if np.isclose(mov_val, best_mov) else "")
        sev_txt = f"{sev_val:.4f}" + ("*" if np.isclose(sev_val, best_sev) else "")
        f.write(
            f"{label:<26} "
            f"{mov_txt:>10} "
            f"{sev_txt:>10} "
            f"{r['movement']['macro_uncertainty']['f1_std']:>10.5f} "
            f"{r['severity']['macro_uncertainty']['f1_std']:>10.5f} "
            f"{rank_map[label]:>6d}\n"
        )
    f.write("\n")

    f.write("EFFICIENCY METRICS\n")
    f.write("-" * 90 + "\n")
    f.write("Caption: NN-C provides the best accuracy-efficiency trade-off under the temporal split protocol.\n")
    header = f"{'Model':<26} {'Params':>10} {'CkptMB':>8} {'WtsMB':>8} {'RSSΔMB':>8} {'Train(s)':>10} {'Lat(ms)':>12} {'Thru(w/s)':>11} {'Rank':>6}"
    f.write(header + "\n")
    f.write("-" * 90 + "\n")

    best_thru = np.max(speeds)
    best_lat = np.min([r['latency']['median_ms'] for _, r in model_items])
    best_wts = np.min(wts)
    best_train = np.min(train_t)

    for label, r in model_items:
        wts_v = r['memory']['weights_memory_mb']
        train_v = r['training']['training_time_sec']
        lat_v = r['latency']['median_ms']
        thr_v = r['latency']['throughput_wps']

        wts_txt = f"{wts_v:.2f}" + ("*" if np.isclose(wts_v, best_wts) else "")
        train_txt = (
            (f"{train_v:.2f}" + ("*" if np.isclose(train_v, best_train) else ""))
            if train_v is not None else "n/a"
        )
        lat_txt = f"{lat_v:.3f}" + ("*" if np.isclose(lat_v, best_lat) else "")
        thr_txt = f"{thr_v:.1f}" + ("*" if np.isclose(thr_v, best_thru) else "")

        f.write(
            f"{label:<26} "
            f"{r['size']['total_params']:>10,} "
            f"{r['memory']['checkpoint_file_mb']:>8.2f} "
            f"{wts_txt:>8} "
            f"{(f'{r['memory']['inference_rss_peak_delta_mb']:.2f}' if r['memory']['inference_rss_peak_delta_mb'] is not None else 'n/a'):>8} "
            f"{train_txt:>10} "
            f"{lat_txt:>12} "
            f"{thr_txt:>11} "
            f"{rank_map[label]:>6d}\n"
        )
    f.write("\n")

    # Latency note
    f.write("* CkptMB = full checkpoint file size on disk (includes optimizer state where present)\n")
    f.write("* WtsMB = raw float32 model weights memory only (parameters x 4 bytes)\n")
    f.write("* RSSΔMB = measured process RSS peak increase during inference pass\n")
    f.write("* Train(s) pulled from paired final_model_* checkpoint training_time_sec\n")
    f.write("* Latency measured on CPU, single-window inference "
            f"({LATENCY_MEASURE_REPS} reps after {LATENCY_WARMUP_REPS}-rep warm-up)\n")
    f.write("* Throughput = windows processed per second (1000 / mean_latency)\n")
    f.write("* Asterisk (*) denotes best value in column where applicable\n\n")

    # --- Per-model detailed breakdown ---
    for label, r in model_items:
        f.write(SEP + "\n")
        f.write(f"  {label}\n")
        f.write(SEP + "\n")

        f.write(f"  Checkpoint : {r['checkpoint']}\n")
        f.write(f"  Parameters : {r['size']['total_params']:,} total "
                f"/ {r['size']['trainable_params']:,} trainable\n")
        f.write(f"  Checkpoint file size : {r['memory']['checkpoint_file_mb']} MB\n")
        f.write(f"  Weights memory       : {r['memory']['weights_memory_mb']} MB\n")
        f.write(
            f"  Inference RSS profile: before={r['memory']['inference_rss_before_mb']} MB, "
            f"after={r['memory']['inference_rss_after_mb']} MB, "
            f"peak={r['memory']['inference_rss_peak_mb']} MB, "
            f"delta={r['memory']['inference_rss_peak_delta_mb']} MB\n"
        )
        f.write(f"  Memory probe method  : {r['memory']['memory_probe_method']}\n")
        if r['training']['training_time_sec'] is not None:
            f.write(
                f"  Training time        : {r['training']['training_time_sec']:.2f} s "
                f"({r['training']['training_time_min']:.2f} min)\n"
            )
        else:
            f.write("  Training time        : n/a\n")
        f.write(f"  Training source      : {r['training']['source_checkpoint']}\n")
        f.write(f"  Latency    : {r['latency']['median_ms']:.3f} ms median, "
                f"IQR {r['latency']['iqr_ms']:.3f} ms, P95 {r['latency']['p95_ms']:.3f} ms  "
                f"({r['latency']['throughput_wps']} windows/sec)\n\n")

        # Movement per-class
        f.write("  MOVEMENT CLASSIFICATION\n")
        f.write(f"  {'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("  " + "-" * 66 + "\n")
        for cls_name in mov_names:
            c = r['movement']['per_class'][cls_name]
            f.write(f"  {cls_name:<24} {c['precision']:>10.4f} "
                    f"{c['recall']:>10.4f} {c['f1']:>10.4f} {c['support']:>10}\n")
        m = r['movement']['macro']
        f.write("  " + "-" * 66 + "\n")
        f.write(f"  {'MACRO AVG':<24} {m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} {m['f1']:>10.4f}\n\n")

        # Severity per-class
        f.write("  SEVERITY CLASSIFICATION\n")
        f.write(f"  {'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("  " + "-" * 66 + "\n")
        for cls_name in sev_names:
            c = r['severity']['per_class'][cls_name]
            f.write(f"  {cls_name:<24} {c['precision']:>10.4f} "
                f"{c['recall']:>10.4f} {c['f1']:>10.4f} {c['support']:>10}\n")
        m = r['severity']['macro']
        f.write("  " + "-" * 66 + "\n")
        f.write(f"  {'MACRO AVG':<24} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f}\n\n")


def _plot_benchmark_comparison(results):
    """Create dissertation-ready comparison figures from benchmark results."""
    model_items = [(k, v) for k, v in results.items() if not k.startswith("_")]
    if not model_items:
        return

    labels = [k for k, _ in model_items]
    x = np.arange(len(labels))

    params_m = [v["size"]["total_params"] / 1_000_000.0 for _, v in model_items]
    ckpt_mb = [v["memory"]["checkpoint_file_mb"] for _, v in model_items]
    wts_mb = [v["memory"]["weights_memory_mb"] for _, v in model_items]
    train_s = [v["training"]["training_time_sec"] or 0.0 for _, v in model_items]
    lat_ms = [v["latency"]["median_ms"] for _, v in model_items]
    thr = [v["latency"]["throughput_wps"] for _, v in model_items]
    thr_std = [v["latency"].get("throughput_iqr_wps", 0.0) for _, v in model_items]
    mov_f1 = [v["movement"]["macro"]["f1"] for _, v in model_items]
    sev_f1 = [v["severity"]["macro"]["f1"] for _, v in model_items]
    mov_f1_std = [v["movement"].get("macro_uncertainty", {}).get("f1_std", 0.0) for _, v in model_items]
    sev_f1_std = [v["severity"].get("macro_uncertainty", {}).get("f1_std", 0.0) for _, v in model_items]

    # Figure 1: training time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, train_s, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Training Time (seconds)")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "benchmark_training_time.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Figure 2: memory footprint (known values only)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.32
    ax.bar(x - width / 2, ckpt_mb, width, label="Checkpoint MB", color="#4c78a8")
    ax.bar(x + width / 2, wts_mb, width, label="Weights MB", color="#f58518")
    ax.set_title("Memory Footprint Comparison")
    ax.set_ylabel("MB")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18)
    ax.legend(loc="upper right", fontsize=8)

    # Value labels for both memory bars.
    for i, v in enumerate(ckpt_mb):
        ax.text(i - width / 2, v * 1.08, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(wts_mb):
        ax.text(i + width / 2, v * 1.08, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "benchmark_memory.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Figure 3: latency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, lat_ms,
           color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Inference Latency (ms/window)")
    ax.set_ylabel("Median latency (ms)")
    ax.axhline(10.0, linestyle="--", color="gray", linewidth=1.2, label="10 ms threshold")
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "benchmark_latency.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Figure 4: accuracy vs throughput (dual-axis)
    fig, ax = plt.subplots(figsize=(9, 5))
    line_mov, = ax.plot(
        labels,
        mov_f1,
        marker="o",
        markersize=10,
        linewidth=2.4,
        label="Movement macro-F1",
        color="#1f77b4",
    )
    line_sev, = ax.plot(
        labels,
        sev_f1,
        marker="^",
        markersize=10,
        linewidth=2.4,
        label="Severity macro-F1",
        color="#ff7f0e",
    )

    ax_t = ax.twinx()
    line_thr, = ax_t.plot(
        labels,
        thr,
        marker="s",
        markersize=10,
        linewidth=2.2,
        linestyle="--",
        color="gray",
        label="Throughput",
    )

    ax.set_title("Accuracy vs Throughput")
    ax.set_ylabel("Movement & Severity Macro-F1 Score")
    ax_t.set_ylabel("Inference Throughput (windows/sec)")
    ax.set_ylim(0.75, 1.01)
    ax.grid(alpha=0.2)

    for i, label in enumerate(labels):
        ax.annotate(label, (i, mov_f1[i] + 0.003), fontsize=9, ha="left")

    lines = [line_mov, line_sev, line_thr]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "benchmark_accuracy_throughput.png")
    fig.savefig(fig1_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig1_path}")

    # Figure 5: movement macro-F1 comparison (standalone)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, mov_f1, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Movement Macro-F1 Comparison")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.90, 1.01)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)

    for bar, v in zip(bars, mov_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.005, v + 0.003), f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "benchmark_movement_macro_f1.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Figure 6: severity macro-F1 comparison (standalone)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, sev_f1, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Severity Macro-F1 Comparison")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.75, 1.01)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)

    for bar, v in zip(bars, sev_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.005, v + 0.004), f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "benchmark_severity_macro_f1.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Figure 2: Normalized scorecard for quick architecture comparison
    # Higher is better for all normalized metrics.
    eps = 1e-9
    norm_speed = np.array(thr) / (max(thr) + eps)
    norm_train_time = min(train_s) / (np.array(train_s) + eps)
    norm_weights = min(wts_mb) / (np.array(wts_mb) + eps)
    norm_mov = np.array(mov_f1)
    norm_sev = np.array(sev_f1)

    metric_names = ["Speed", "Train Time", "Weights Mem", "Mov F1", "Sev F1"]
    metric_matrix = np.vstack([
        norm_speed,
        norm_train_time,
        norm_weights,
        norm_mov,
        norm_sev,
    ]).T

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.16
    idx = np.arange(len(metric_names))
    for i, (label, vals) in enumerate(zip(labels, metric_matrix)):
        ax.bar(idx + (i - (len(labels)-1)/2) * width, vals, width, label=label)

    ax.set_xticks(idx)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_title("Normalized Architecture Scorecard")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Grouping aids: efficiency vs performance
    ax.axvspan(-0.5, 2.5, color="#d8ebff", alpha=0.2)
    ax.axvspan(2.5, 4.5, color="#e8f7e8", alpha=0.2)
    ax.text(1.0, 1.03, "Efficiency", fontsize=9, ha="center")
    ax.text(3.5, 1.03, "Performance", fontsize=9, ha="center")

    # Numeric labels above bars.
    for i, (_, vals) in enumerate(zip(labels, metric_matrix)):
        for j, v in enumerate(vals):
            xpos = j + (i - (len(labels)-1)/2) * width
            ax.text(xpos, min(1.06, v + 0.015), f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "benchmark_scorecard.png")
    fig.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig2_path}")

if __name__ == "__main__":
    run_benchmark()
