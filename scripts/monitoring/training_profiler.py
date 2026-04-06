"""
training_profiler.py
====================
High-frequency hardware + training profiler.

Samples CPU, GPU, RAM, disk I/O, and training iteration state every N seconds
(default: 0.5s) and writes timestamped CSV + live console summary.

After profiling, generates correlation analysis and bottleneck diagnosis.

Usage:
    # Profile for 5 minutes at 0.5s intervals (default):
    python scripts/monitoring/training_profiler.py

    # Profile for 10 minutes at 1s intervals:
    python scripts/monitoring/training_profiler.py --duration 600 --interval 1.0

    # Just analyze an existing CSV:
    python scripts/monitoring/training_profiler.py --analyze profiling_results/profile_XXXX.csv

Outputs:
    profiling_results/profile_<timestamp>.csv   — raw data
    profiling_results/profile_<timestamp>.png   — correlation heatmap + timeline
    profiling_results/profile_<timestamp>.txt   — bottleneck analysis summary
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = REPO_ROOT / "profiling_results"
TRAINING_LOG_BASE = Path(
    os.environ.get("nnUNet_results", REPO_ROOT / "data" / "nnUNet_results")
)

DEFAULT_DURATION = 300   # seconds
DEFAULT_INTERVAL = 0.5   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# System metrics collection
# ─────────────────────────────────────────────────────────────────────────────

def _get_cpu_times():
    """Get per-CPU idle+total times via ctypes (no psutil needed)."""
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        def ft_to_int(ft):
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        
        idle_t = ft_to_int(idle)
        total_t = ft_to_int(kernel) + ft_to_int(user)
        return idle_t, total_t
    except Exception:
        return None, None


def get_cpu_percent(prev_idle, prev_total):
    """Calculate CPU usage % since last call."""
    idle, total = _get_cpu_times()
    if idle is None or prev_idle is None:
        return 0.0, idle, total
    d_idle = idle - prev_idle
    d_total = total - prev_total
    if d_total == 0:
        return 0.0, idle, total
    cpu_pct = (1.0 - d_idle / d_total) * 100.0
    return max(0.0, min(100.0, cpu_pct)), idle, total


def get_ram_usage():
    """Get RAM usage in MB and percent via kernel32."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_mb = stat.ullTotalPhys / (1024 * 1024)
        used_mb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024)
        pct = stat.dwMemoryLoad
        return used_mb, total_mb, pct
    except Exception:
        return 0, 0, 0


def get_gpu_metrics():
    """Query GPU via nvidia-smi CSV output."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu="
                "utilization.gpu,"
                "utilization.memory,"
                "memory.used,"
                "memory.total,"
                "temperature.gpu,"
                "power.draw,"
                "power.limit,"
                "clocks.current.graphics,"
                "clocks.current.memory,"
                "clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {}
        
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 9:
            return {}
        
        return {
            "gpu_util_pct": float(parts[0]) if parts[0] not in ("[N/A]", "") else 0,
            "gpu_mem_util_pct": float(parts[1]) if parts[1] not in ("[N/A]", "") else 0,
            "gpu_mem_used_mb": float(parts[2]) if parts[2] not in ("[N/A]", "") else 0,
            "gpu_mem_total_mb": float(parts[3]) if parts[3] not in ("[N/A]", "") else 0,
            "gpu_temp_c": float(parts[4]) if parts[4] not in ("[N/A]", "") else 0,
            "gpu_power_w": float(parts[5]) if parts[5] not in ("[N/A]", "") else 0,
            "gpu_power_limit_w": float(parts[6]) if parts[6] not in ("[N/A]", "") else 0,
            "gpu_clock_mhz": float(parts[7]) if parts[7] not in ("[N/A]", "") else 0,
            "gpu_mem_clock_mhz": float(parts[8]) if parts[8] not in ("[N/A]", "") else 0,
            "gpu_throttle": parts[9].strip() if len(parts) > 9 else "",
        }
    except Exception as e:
        return {"gpu_error": str(e)}


def get_training_process_info():
    """Get training process CPU%, RAM, thread count via WMI."""
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "$p = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                "| Where-Object {$_.CommandLine -like '*nnUNetv2_train*'} "
                "| Select-Object -First 1; "
                "if ($p) { "
                "  $proc = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue; "
                "  \"$($p.ProcessId),$($p.ThreadCount),"
                "$([math]::Round($proc.WorkingSet64/1MB,1)),"
                "$([math]::Round($proc.CPU,1))\" "
                "} else { 'none' }"
            ],
            capture_output=True, text=True, timeout=10
        )
        line = result.stdout.strip()
        if line == "none" or not line:
            return {}
        parts = line.split(",")
        return {
            "train_pid": int(parts[0]),
            "train_threads": int(parts[1]),
            "train_ram_mb": float(parts[2]),
            "train_cpu_sec": float(parts[3]),
        }
    except Exception:
        return {}


def get_training_iteration():
    """
    Read the training log to determine current epoch/iteration state.
    Returns dict with epoch, phase (train/val), and the latest metrics.
    """
    try:
        # Find the latest training log
        log_files = sorted(TRAINING_LOG_BASE.rglob("training_log_*.txt"))
        if not log_files:
            return {}
        log_file = log_files[-1]
        
        # Read last 30 lines efficiently
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        tail = lines[-30:] if len(lines) > 30 else lines
        
        info = {}
        for line in tail:
            line = line.strip()
            if "Epoch " in line and "time" not in line.lower():
                # e.g. "2026-03-05 22:49:06.352011: Epoch 0"
                try:
                    epoch_str = line.split("Epoch")[-1].strip()
                    info["epoch"] = int(epoch_str)
                    info["phase"] = "train"  # Epoch start = training begins
                except (ValueError, IndexError):
                    pass
            elif "train_loss" in line:
                try:
                    info["train_loss"] = float(line.split("train_loss")[-1].strip())
                    info["phase"] = "val"  # train done, validation running
                except (ValueError, IndexError):
                    pass
            elif "val_loss" in line:
                try:
                    info["val_loss"] = float(line.split("val_loss")[-1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Pseudo dice" in line:
                try:
                    import re
                    m = re.search(r'(\d+\.\d+)', line)
                    if m:
                        info["pseudo_dice"] = float(m.group(1))
                except (ValueError, IndexError):
                    pass
            elif "Epoch time" in line:
                try:
                    info["epoch_time_s"] = float(
                        line.split("Epoch time:")[-1].strip().replace(" s", "")
                    )
                    info["phase"] = "checkpoint"  # Epoch finished, saving
                except (ValueError, IndexError):
                    pass
            elif "Current learning rate" in line:
                info["phase"] = "train"
        
        # Also track log file modification time to detect staleness
        info["log_age_s"] = time.time() - log_file.stat().st_mtime
        info["log_file"] = log_file.name
        
        return info
    except Exception:
        return {}


def get_per_core_cpu():
    """Get per-core CPU usage via a quick PowerShell query."""
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
                "| Where-Object {$_.Name -ne '_Total'} "
                "| ForEach-Object { $_.PercentProcessorTime }) -join ','"
            ],
            capture_output=True, text=True, timeout=10
        )
        vals = result.stdout.strip()
        if vals:
            return [float(v) for v in vals.split(",") if v.strip()]
        return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Profiler loop
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "timestamp", "elapsed_s",
    # CPU
    "cpu_pct", "cpu_core_max_pct",
    # RAM
    "ram_used_mb", "ram_total_mb", "ram_pct",
    # GPU  
    "gpu_util_pct", "gpu_mem_util_pct",
    "gpu_mem_used_mb", "gpu_mem_total_mb",
    "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w",
    "gpu_clock_mhz", "gpu_mem_clock_mhz", "gpu_throttle",
    # Training process
    "train_pid", "train_threads", "train_ram_mb", "train_cpu_sec",
    # Training state
    "epoch", "phase", "train_loss", "val_loss", "pseudo_dice",
    "epoch_time_s", "log_age_s",
]


_stop = False

def _signal_handler(sig, frame):
    global _stop
    _stop = True


def run_profiler(duration: float, interval: float, csv_path: Path):
    """Main profiling loop."""
    global _stop
    signal.signal(signal.SIGINT, _signal_handler)
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Training Profiler — sampling every {interval}s for {duration}s      ║")
    print(f"║  Output: {csv_path.name:<50s}║")
    print(f"║  Press Ctrl+C to stop early                                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        
        start = time.time()
        sample_count = 0
        prev_idle, prev_total = _get_cpu_times()
        
        # Header for live display
        print(f"{'Time':>8s}  {'CPU%':>5s}  {'CoreMax':>7s}  {'RAM_MB':>7s}  "
              f"{'GPU%':>5s}  {'VRAM_MB':>8s}  {'Power':>6s}  {'Clock':>6s}  "
              f"{'Epoch':>5s}  {'Phase':>10s}  {'TrainRAM':>8s}  {'Threads':>7s}")
        print("─" * 110)
        
        while (time.time() - start) < duration and not _stop:
            t0 = time.time()
            elapsed = t0 - start
            
            # Collect all metrics
            cpu_pct, prev_idle, prev_total = get_cpu_percent(prev_idle, prev_total)
            ram_used, ram_total, ram_pct = get_ram_usage()
            gpu = get_gpu_metrics()
            proc = get_training_process_info()
            train_state = get_training_iteration()
            
            # Per-core CPU (every 5th sample to reduce overhead)
            core_max = 0
            if sample_count % 5 == 0:
                cores = get_per_core_cpu()
                if cores:
                    core_max = max(cores)
            
            row = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "elapsed_s": round(elapsed, 2),
                "cpu_pct": round(cpu_pct, 1),
                "cpu_core_max_pct": round(core_max, 1),
                "ram_used_mb": round(ram_used, 0),
                "ram_total_mb": round(ram_total, 0),
                "ram_pct": ram_pct,
                **{k: v for k, v in gpu.items() if k in FIELDNAMES},
                **{k: v for k, v in proc.items() if k in FIELDNAMES},
                **{k: v for k, v in train_state.items() if k in FIELDNAMES},
            }
            
            writer.writerow(row)
            f.flush()
            sample_count += 1
            
            # Live console output
            epoch_str = str(row.get("epoch", "?"))
            phase_str = str(row.get("phase", "?"))[:10]
            train_ram = row.get("train_ram_mb", 0)
            threads = row.get("train_threads", 0)
            gpu_pct = row.get("gpu_util_pct", 0)
            vram = row.get("gpu_mem_used_mb", 0)
            power = row.get("gpu_power_w", 0)
            clock = row.get("gpu_clock_mhz", 0)
            
            print(f"\r{elapsed:7.1f}s  {cpu_pct:5.1f}  {core_max:7.1f}  "
                  f"{ram_used:7.0f}  {gpu_pct:5.1f}  {vram:8.0f}  "
                  f"{power:5.0f}W  {clock:5.0f}M  "
                  f"{epoch_str:>5s}  {phase_str:<10s}  {train_ram:7.0f}M  "
                  f"{threads:>7d}", end="")
            
            # Sleep for remaining interval
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
    
    print(f"\n\nProfiler stopped. {sample_count} samples in {time.time()-start:.1f}s")
    print(f"Data saved to: {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(csv_path: Path):
    """Load CSV, compute stats and correlations, generate plots + summary."""
    import numpy as np
    
    # Load data
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(val)
    
    if not data or "elapsed_s" not in data:
        print("No data to analyze.")
        return
    
    # Convert numeric columns
    numeric_cols = [
        "elapsed_s", "cpu_pct", "cpu_core_max_pct",
        "ram_used_mb", "ram_pct",
        "gpu_util_pct", "gpu_mem_util_pct",
        "gpu_mem_used_mb", "gpu_temp_c", "gpu_power_w",
        "gpu_clock_mhz", "gpu_mem_clock_mhz",
        "train_ram_mb", "train_threads", "train_cpu_sec",
        "log_age_s",
    ]
    
    arrays = {}
    for col in numeric_cols:
        if col in data:
            vals = []
            for v in data[col]:
                try:
                    vals.append(float(v) if v else np.nan)
                except (ValueError, TypeError):
                    vals.append(np.nan)
            arr = np.array(vals)
            if not np.all(np.isnan(arr)):
                arrays[col] = arr
    
    n = len(data.get("elapsed_s", []))
    
    # ── Summary stats ──
    summary_path = csv_path.with_suffix(".txt")
    lines = []
    lines.append("=" * 70)
    lines.append(f"TRAINING PROFILER ANALYSIS — {n} samples")
    lines.append(f"Source: {csv_path.name}")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("─── Summary Statistics ───")
    lines.append(f"{'Metric':<25s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  "
                 f"{'Max':>8s}  {'P5':>8s}  {'P95':>8s}")
    lines.append("─" * 85)
    
    for col in ["cpu_pct", "cpu_core_max_pct", "ram_used_mb", "ram_pct",
                "gpu_util_pct", "gpu_mem_used_mb", "gpu_temp_c", "gpu_power_w",
                "gpu_clock_mhz", "train_ram_mb", "train_threads"]:
        if col not in arrays:
            continue
        a = arrays[col]
        valid = a[~np.isnan(a)]
        if len(valid) == 0:
            continue
        lines.append(
            f"{col:<25s}  {np.mean(valid):8.1f}  {np.std(valid):8.1f}  "
            f"{np.min(valid):8.1f}  {np.max(valid):8.1f}  "
            f"{np.percentile(valid, 5):8.1f}  {np.percentile(valid, 95):8.1f}"
        )
    
    lines.append("")
    
    # ── GPU utilization distribution ──
    if "gpu_util_pct" in arrays:
        gpu = arrays["gpu_util_pct"]
        valid_gpu = gpu[~np.isnan(gpu)]
        lines.append("─── GPU Utilization Distribution ───")
        for threshold in [0, 10, 25, 50, 75, 90, 100]:
            count = np.sum(valid_gpu <= threshold)
            pct = count / len(valid_gpu) * 100
            lines.append(f"  GPU ≤ {threshold:3d}%:  {pct:5.1f}% of samples ({count}/{len(valid_gpu)})")
        lines.append(f"  → GPU is idle (≤10%) {np.sum(valid_gpu <= 10)/len(valid_gpu)*100:.0f}% of the time")
        lines.append("")
    
    # ── Phase analysis ──
    if "phase" in data and "gpu_util_pct" in arrays:
        lines.append("─── GPU Usage by Training Phase ───")
        phases = data["phase"]
        for phase_name in ["train", "val", "checkpoint"]:
            mask = np.array([p == phase_name for p in phases])
            if np.sum(mask) > 0:
                phase_gpu = arrays["gpu_util_pct"][mask]
                phase_cpu = arrays["cpu_pct"][mask] if "cpu_pct" in arrays else np.array([0])
                valid_gpu = phase_gpu[~np.isnan(phase_gpu)]
                valid_cpu = phase_cpu[~np.isnan(phase_cpu)]
                if len(valid_gpu) > 0:
                    lines.append(
                        f"  {phase_name:<12s}: GPU {np.mean(valid_gpu):5.1f}% (±{np.std(valid_gpu):.1f}), "
                        f"CPU {np.mean(valid_cpu):5.1f}% (±{np.std(valid_cpu):.1f}), "
                        f"{np.sum(mask)} samples"
                    )
        lines.append("")
    
    # ── Correlation matrix ──
    corr_cols = ["cpu_pct", "gpu_util_pct", "gpu_power_w", "gpu_clock_mhz",
                 "gpu_mem_used_mb", "ram_used_mb", "train_ram_mb"]
    available_cols = [c for c in corr_cols if c in arrays]
    
    if len(available_cols) >= 2:
        lines.append("─── Correlation Matrix (Pearson) ───")
        # Build matrix
        matrix = np.column_stack([arrays[c] for c in available_cols])
        # Remove rows with NaN
        mask = ~np.any(np.isnan(matrix), axis=1)
        matrix = matrix[mask]
        
        if len(matrix) > 10:
            corr = np.corrcoef(matrix.T)
            header = f"{'':>18s}  " + "  ".join(f"{c[:8]:>8s}" for c in available_cols)
            lines.append(header)
            for i, col in enumerate(available_cols):
                row_str = f"{col:<18s}  " + "  ".join(
                    f"{corr[i, j]:8.3f}" for j in range(len(available_cols))
                )
                lines.append(row_str)
        lines.append("")
    
    # ── Bottleneck diagnosis ──
    lines.append("─── Bottleneck Diagnosis ───")
    
    if "gpu_util_pct" in arrays and "cpu_pct" in arrays:
        gpu_mean = np.nanmean(arrays["gpu_util_pct"])
        cpu_mean = np.nanmean(arrays["cpu_pct"])
        gpu_idle_pct = np.sum(arrays["gpu_util_pct"][~np.isnan(arrays["gpu_util_pct"])] <= 10) / \
                       np.sum(~np.isnan(arrays["gpu_util_pct"])) * 100
        
        if gpu_idle_pct > 50 and cpu_mean < 60:
            lines.append(f"  ⚠ GPU idle {gpu_idle_pct:.0f}% of time, CPU at {cpu_mean:.0f}%")
            lines.append(f"    → DATA PIPELINE BOTTLENECK (Python GIL or I/O)")
            lines.append(f"    → CPU is not maxed → GIL contention between augmentation threads")
            lines.append(f"    → Consider: multiprocessing (if stable), reducing augmentation,")
            lines.append(f"      or pre-computing augmented batches to disk")
        elif gpu_idle_pct > 50 and cpu_mean > 80:
            lines.append(f"  ⚠ GPU idle {gpu_idle_pct:.0f}% of time, CPU at {cpu_mean:.0f}%")
            lines.append(f"    → CPU BOTTLENECK — augmentation threads saturating all cores")
            lines.append(f"    → Consider: lighter augmentations, GPU-based augmentation,")
            lines.append(f"      or pre-augmenting to disk")
        elif gpu_mean > 80:
            lines.append(f"  ✓ GPU well utilized ({gpu_mean:.0f}%). Training is GPU-bound.")
        else:
            lines.append(f"  GPU avg: {gpu_mean:.0f}%, CPU avg: {cpu_mean:.0f}%")
            lines.append(f"  GPU idle {gpu_idle_pct:.0f}% of the time")
    
    if "gpu_power_w" in arrays and "gpu_power_limit_w" in arrays:
        power_ratio = np.nanmean(arrays["gpu_power_w"]) / np.nanmean(arrays["gpu_power_limit_w"])
        if power_ratio < 0.4:
            lines.append(f"  ⚠ GPU power draw {power_ratio*100:.0f}% of TDP → GPU underworked")
    
    if "gpu_clock_mhz" in arrays:
        clock = arrays["gpu_clock_mhz"]
        valid_clock = clock[~np.isnan(clock)]
        if len(valid_clock) > 0:
            clock_range = np.max(valid_clock) - np.min(valid_clock)
            if clock_range > 500:
                lines.append(f"  ⚠ GPU clock varies widely ({np.min(valid_clock):.0f}-"
                             f"{np.max(valid_clock):.0f} MHz) → throttling or idle downclocking")
    
    lines.append("")
    
    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)
    print(f"\nSummary saved to: {summary_path}")
    
    # ── Plot ──
    try:
        _generate_plots(csv_path, arrays, data, available_cols)
    except Exception as e:
        print(f"Plot generation failed: {e}")


def _generate_plots(csv_path: Path, arrays: dict, data: dict, corr_cols: list):
    """Generate timeline + correlation heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    
    elapsed = arrays.get("elapsed_s", np.array([]))
    if len(elapsed) == 0:
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
    fig.suptitle("Training Profiler — Hardware Utilization Timeline", fontsize=14, y=0.98)
    
    # 1) CPU + GPU utilization
    ax = axes[0]
    if "cpu_pct" in arrays:
        ax.plot(elapsed, arrays["cpu_pct"], label="CPU %", color="blue", alpha=0.7, linewidth=0.8)
    if "gpu_util_pct" in arrays:
        ax.plot(elapsed, arrays["gpu_util_pct"], label="GPU %", color="red", alpha=0.7, linewidth=0.8)
    ax.set_ylabel("Utilization %")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right")
    ax.set_title("CPU & GPU Utilization")
    ax.grid(True, alpha=0.3)
    
    # Color background by phase
    if "phase" in data:
        phases = data["phase"]
        for i in range(len(phases) - 1):
            if phases[i] == "train":
                ax.axvspan(elapsed[i], elapsed[i+1], alpha=0.05, color="green")
            elif phases[i] == "val":
                ax.axvspan(elapsed[i], elapsed[i+1], alpha=0.05, color="orange")
    
    # 2) Memory
    ax = axes[1]
    if "gpu_mem_used_mb" in arrays:
        ax.plot(elapsed, arrays["gpu_mem_used_mb"], label="VRAM (MB)", color="red", linewidth=0.8)
    if "train_ram_mb" in arrays:
        ax.plot(elapsed, arrays["train_ram_mb"], label="Process RAM (MB)", color="blue", linewidth=0.8)
    ax.set_ylabel("Memory (MB)")
    ax.legend(loc="upper right")
    ax.set_title("Memory Usage")
    ax.grid(True, alpha=0.3)
    
    # 3) GPU Power + Clock
    ax = axes[2]
    if "gpu_power_w" in arrays:
        ax.plot(elapsed, arrays["gpu_power_w"], label="Power (W)", color="orange", linewidth=0.8)
        ax.set_ylabel("Power (W)", color="orange")
    ax2 = ax.twinx()
    if "gpu_clock_mhz" in arrays:
        ax2.plot(elapsed, arrays["gpu_clock_mhz"], label="GPU Clock (MHz)", color="green", linewidth=0.8)
        ax2.set_ylabel("Clock (MHz)", color="green")
    ax.set_title("GPU Power & Clock")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    # 4) Correlation heatmap
    ax = axes[3]
    if len(corr_cols) >= 2:
        matrix = np.column_stack([arrays[c] for c in corr_cols])
        mask = ~np.any(np.isnan(matrix), axis=1)
        matrix = matrix[mask]
        if len(matrix) > 10:
            corr = np.corrcoef(matrix.T)
            short_names = [c.replace("gpu_", "G_").replace("train_", "T_").replace("_pct", "%")
                           .replace("_mb", "")[:12] for c in corr_cols]
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr_cols)))
            ax.set_yticks(range(len(corr_cols)))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(short_names, fontsize=9)
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(corr[i,j]) > 0.5 else "black")
            fig.colorbar(im, ax=ax, shrink=0.6)
            ax.set_title("Correlation Matrix")
    
    for ax in axes[:3]:
        ax.set_xlabel("Elapsed (s)")
    
    plt.tight_layout()
    plot_path = csv_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training hardware profiler")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"Profiling duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Sampling interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to existing CSV to analyze (skip profiling)")
    args = parser.parse_args()
    
    if args.analyze:
        analyze(Path(args.analyze))
        return
    
    # Generate output path
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"profile_{ts}.csv"
    
    # Run profiler
    csv_path = run_profiler(args.duration, args.interval, csv_path)
    
    # Analyze
    print("\n" + "=" * 60)
    print("Analyzing collected data...")
    print("=" * 60 + "\n")
    analyze(csv_path)


if __name__ == "__main__":
    main()
