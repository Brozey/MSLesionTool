"""
utils/thesis_logger.py
======================
Thesis-grade logging infrastructure for the FLAIR lesion segmentation pipeline.

Provides:
    - **Hardware profiling**: GPU model, VRAM, CUDA version, CPU, RAM, OS,
      Python version, PyTorch version, nnU-Net version.
    - **Phase/experiment timing**: context managers and decorators for automatic
      wall-clock timing of every pipeline stage.
    - **Structured log output**: dual console + rotating file logger, plus a
      JSON hardware manifest and a CSV experiment-timing ledger.
    - **Peak GPU memory tracking**: records high-water-mark VRAM usage per
      experiment for reporting in the thesis.

Usage
-----
    from utils.thesis_logger import (
        init_thesis_logging,
        log_hardware_summary,
        PhaseTimer,
        ExperimentTracker,
    )

    init_thesis_logging(log_dir="logs")
    log_hardware_summary()

    tracker = ExperimentTracker(log_dir="logs")
    with tracker.experiment("DS500_CNN_3d_fullres"):
        ...  # training code

    tracker.save_summary()
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import platform
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nnunet_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Hardware information collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_hardware_info() -> Dict[str, Any]:
    """
    Gather a comprehensive hardware + software snapshot.

    Returns a dict suitable for JSON serialisation and logging.
    """
    info: Dict[str, Any] = {}

    # ── Operating System ─────────────────────────────────────────────────
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
    }

    # ── Python ───────────────────────────────────────────────────────────
    info["python"] = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }

    # ── CPU ──────────────────────────────────────────────────────────────
    cpu_info: Dict[str, Any] = {
        "physical_cores": None,
        "logical_cores": None,
        "processor": platform.processor(),
    }
    try:
        import psutil
        cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
        cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
    except ImportError:
        cpu_info["physical_cores"] = os.cpu_count()
        cpu_info["logical_cores"] = os.cpu_count()

    # Try to get CPU brand string on Windows
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_info["brand"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            cpu_info["brand"] = cpu_info["processor"]
    else:
        cpu_info["brand"] = cpu_info["processor"]

    info["cpu"] = cpu_info

    # ── RAM ──────────────────────────────────────────────────────────────
    ram_info: Dict[str, Any] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_info["total_gb"] = round(vm.total / (1024 ** 3), 2)
        ram_info["available_gb"] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        ram_info["total_gb"] = "psutil not installed"
    info["ram"] = ram_info

    # ── GPU / CUDA ───────────────────────────────────────────────────────
    gpu_info: Dict[str, Any] = {"available": False}
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["pytorch_version"] = torch.__version__
        gpu_info["cuda_version_runtime"] = (
            torch.version.cuda if torch.version.cuda else "N/A"
        )
        gpu_info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        gpu_info["cudnn_enabled"] = torch.backends.cudnn.enabled

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                })
            gpu_info["devices"] = devices
    except ImportError:
        gpu_info["pytorch_version"] = "NOT INSTALLED"

    info["gpu"] = gpu_info

    # ── nnU-Net version ──────────────────────────────────────────────────
    try:
        import nnunetv2
        info["nnunet_version"] = getattr(nnunetv2, "__version__", "installed (version unknown)")
    except ImportError:
        info["nnunet_version"] = "NOT INSTALLED"

    # ── Key library versions ─────────────────────────────────────────────
    lib_versions: Dict[str, str] = {}
    for lib_name in ("numpy", "scipy", "nibabel", "matplotlib",
                     "scikit-learn", "pandas", "psutil"):
        try:
            mod = __import__(lib_name.replace("-", "_"))
            lib_versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            lib_versions[lib_name] = "not installed"
    # pyyaml imports as 'yaml'
    try:
        import yaml
        lib_versions["pyyaml"] = getattr(yaml, "__version__", "unknown")
    except ImportError:
        lib_versions["pyyaml"] = "not installed"
    info["libraries"] = lib_versions

    # ── Timestamp ────────────────────────────────────────────────────────
    info["timestamp"] = datetime.datetime.now().isoformat()

    return info


def log_hardware_summary(log_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Collect hardware info, log it to console, and optionally save to JSON.

    Parameters
    ----------
    log_dir : If provided, writes ``hardware_manifest.json`` into this
              directory.

    Returns
    -------
    The hardware info dict.
    """
    info = collect_hardware_info()

    logger.info("=" * 72)
    logger.info("  HARDWARE & SOFTWARE ENVIRONMENT")
    logger.info("=" * 72)
    logger.info("  Timestamp       : %s", info["timestamp"])
    logger.info("  Hostname        : %s", info["os"]["hostname"])
    logger.info("  OS              : %s %s (%s)",
                info["os"]["system"], info["os"]["release"], info["os"]["machine"])
    logger.info("  Python          : %s (%s)",
                info["python"]["version"], info["python"]["implementation"])
    logger.info("  CPU             : %s", info["cpu"].get("brand", info["cpu"]["processor"]))
    logger.info("  CPU cores       : %s physical / %s logical",
                info["cpu"]["physical_cores"], info["cpu"]["logical_cores"])

    ram = info["ram"]
    if isinstance(ram.get("total_gb"), (int, float)):
        logger.info("  RAM             : %.1f GB total / %.1f GB available",
                    ram["total_gb"], ram.get("available_gb", 0))
    else:
        logger.info("  RAM             : %s", ram.get("total_gb", "unknown"))

    gpu = info["gpu"]
    logger.info("  PyTorch         : %s", gpu.get("pytorch_version", "N/A"))
    logger.info("  CUDA runtime    : %s", gpu.get("cuda_version_runtime", "N/A"))
    logger.info("  cuDNN           : %s (enabled=%s)",
                gpu.get("cudnn_version", "N/A"), gpu.get("cudnn_enabled", "N/A"))

    if gpu.get("available"):
        for dev in gpu.get("devices", []):
            logger.info("  GPU %d           : %s  (%.1f GB VRAM, SM %d.%d, %d SMs)",
                        dev["index"], dev["name"], dev["total_memory_gb"],
                        dev["major"], dev["minor"], dev["multi_processor_count"])
    else:
        logger.info("  GPU             : NOT AVAILABLE")

    logger.info("  nnU-Net         : %s", info.get("nnunet_version", "N/A"))
    logger.info("  Key libraries   : %s",
                ", ".join(f"{k}={v}" for k, v in info.get("libraries", {}).items()))
    logger.info("=" * 72)

    # Save JSON manifest
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = log_dir / "hardware_manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(info, fh, indent=2, default=str)
        logger.info("Hardware manifest saved to %s", manifest_path)

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ──────────────────────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS.s string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


@contextmanager
def PhaseTimer(phase_name: str):
    """
    Context manager that logs the start/end and elapsed wall-clock time
    of a named pipeline phase.

    Usage::

        with PhaseTimer("Phase 2: Training"):
            ...  # training code

    Yields a dict ``{"start": float, "end": float, "elapsed": float}``
    that is populated on exit.
    """
    timing: Dict[str, float] = {}
    start_dt = datetime.datetime.now()

    logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    logger.info("┃  START: %-52s ┃", phase_name)
    logger.info("┃  Time : %-52s ┃", start_dt.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

    timing["start"] = time.time()
    timing["start_iso"] = start_dt.isoformat()

    try:
        yield timing
    finally:
        timing["end"] = time.time()
        end_dt = datetime.datetime.now()
        timing["end_iso"] = end_dt.isoformat()
        timing["elapsed"] = timing["end"] - timing["start"]

        # GPU memory peak (if available)
        peak_gpu_mb = _get_peak_gpu_memory()
        if peak_gpu_mb is not None:
            timing["peak_gpu_memory_mb"] = peak_gpu_mb

        logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        logger.info("┃  DONE : %-52s ┃", phase_name)
        logger.info("┃  Time : %-52s ┃",
                    _format_duration(timing["elapsed"]))
        if peak_gpu_mb is not None:
            logger.info("┃  Peak GPU memory: %-42s ┃",
                        f"{peak_gpu_mb:.0f} MB ({peak_gpu_mb / 1024:.2f} GB)")
        logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")


def _get_peak_gpu_memory() -> Optional[float]:
    """Return peak GPU memory usage in MB, or None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            return peak_bytes / (1024 ** 2)
    except ImportError:
        pass
    return None


def _reset_gpu_memory_stats() -> None:
    """Reset PyTorch CUDA memory tracking counters."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Experiment tracker
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Track timing and GPU memory for multiple experiments and save a
    consolidated CSV + JSON summary.

    Usage::

        tracker = ExperimentTracker(log_dir=Path("logs"))

        with tracker.experiment("DS500_CNN"):
            ...  # train experiment 1

        with tracker.experiment("DS500_ResEncL"):
            ...  # train experiment 2

        tracker.save_summary()
    """

    def __init__(self, log_dir: Path | str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []
        self._pipeline_start = time.time()
        self._pipeline_start_iso = datetime.datetime.now().isoformat()

    @contextmanager
    def experiment(self, name: str):
        """Context manager for a named experiment / pipeline phase."""
        _reset_gpu_memory_stats()

        record: Dict[str, Any] = {"name": name}
        start = time.time()
        start_dt = datetime.datetime.now()
        record["start_iso"] = start_dt.isoformat()

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info("║  EXPERIMENT: %-47s ║", name)
        logger.info("║  Started   : %-47s ║", start_dt.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("╚══════════════════════════════════════════════════════════════╝")

        try:
            yield record
        except Exception as exc:
            record["status"] = "FAILED"
            record["error"] = str(exc)
            raise
        finally:
            end = time.time()
            end_dt = datetime.datetime.now()
            elapsed = end - start
            record["end_iso"] = end_dt.isoformat()
            record["elapsed_seconds"] = round(elapsed, 2)
            record["elapsed_formatted"] = _format_duration(elapsed)
            record.setdefault("status", "OK")

            peak = _get_peak_gpu_memory()
            if peak is not None:
                record["peak_gpu_memory_mb"] = round(peak, 1)
                record["peak_gpu_memory_gb"] = round(peak / 1024, 2)

            self._records.append(record)

            logger.info("")
            logger.info("╔══════════════════════════════════════════════════════════════╗")
            logger.info("║  FINISHED  : %-47s ║", name)
            logger.info("║  Duration  : %-47s ║", record["elapsed_formatted"])
            logger.info("║  Status    : %-47s ║", record["status"])
            if peak is not None:
                logger.info("║  Peak VRAM : %-47s ║",
                            f"{peak:.0f} MB ({peak / 1024:.2f} GB)")
            logger.info("╚══════════════════════════════════════════════════════════════╝")

    def save_summary(self) -> None:
        """Write experiment timing summary to CSV and JSON."""
        total_elapsed = time.time() - self._pipeline_start

        # ── JSON ─────────────────────────────────────────────────────────
        summary = {
            "pipeline_start": self._pipeline_start_iso,
            "pipeline_end": datetime.datetime.now().isoformat(),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_formatted": _format_duration(total_elapsed),
            "experiments": self._records,
        }
        json_path = self.log_dir / "experiment_timings.json"
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Experiment timings (JSON) saved to %s", json_path)

        # ── CSV ──────────────────────────────────────────────────────────
        csv_path = self.log_dir / "experiment_timings.csv"
        if self._records:
            fieldnames = [
                "name", "status", "start_iso", "end_iso",
                "elapsed_seconds", "elapsed_formatted",
                "peak_gpu_memory_mb", "peak_gpu_memory_gb",
            ]
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._records)
            logger.info("Experiment timings (CSV)  saved to %s", csv_path)

        # ── Console summary table ────────────────────────────────────────
        self._print_summary_table(total_elapsed)

    def _print_summary_table(self, total_elapsed: float) -> None:
        """Print a formatted summary table to the log."""
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════════════════╗")
        logger.info("║                      PIPELINE EXECUTION SUMMARY                         ║")
        logger.info("╠══════════════════════════════════════════════════════════════════════════╣")
        logger.info("║  %-35s %8s  %8s  %10s  ║",
                    "Experiment", "Status", "Duration", "Peak VRAM")
        logger.info("║  %-35s %8s  %8s  %10s  ║",
                    "─" * 35, "─" * 8, "─" * 8, "─" * 10)

        for rec in self._records:
            vram_str = (f"{rec.get('peak_gpu_memory_gb', 0):.1f} GB"
                        if "peak_gpu_memory_gb" in rec else "N/A")
            logger.info("║  %-35s %8s  %8s  %10s  ║",
                        rec["name"][:35],
                        rec["status"],
                        rec["elapsed_formatted"],
                        vram_str)

        logger.info("║  %-35s %8s  %8s  %10s  ║",
                    "─" * 35, "─" * 8, "─" * 8, "─" * 10)
        logger.info("║  %-35s %8s  %8s  %10s  ║",
                    "TOTAL PIPELINE TIME", "",
                    _format_duration(total_elapsed), "")
        logger.info("╚══════════════════════════════════════════════════════════════════════════╝")


# ──────────────────────────────────────────────────────────────────────────────
# Logging initialisation (enhanced for thesis)
# ──────────────────────────────────────────────────────────────────────────────

def init_thesis_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> Path:
    """
    Set up dual-output logging (console + timestamped log file) and
    immediately log the hardware manifest.

    Parameters
    ----------
    log_dir : Directory where log files and manifests are written.
    level   : Logging level (default: INFO).
    console : Whether to also print to stdout.

    Returns
    -------
    Path to the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{ts}.log"

    # Clear any existing handlers on the pipeline logger
    root_logger = logging.getLogger("nnunet_pipeline")
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (always)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # Console handler (optional) — force UTF-8 to support box-drawing chars
    if console:
        ch = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                 closefd=False, buffering=1)
        )
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    # Also capture warnings via the logging system
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.addHandler(fh)
    if console:
        warn_logger.addHandler(ch)

    root_logger.info("Logging initialised -> %s", log_file)

    # Log hardware summary automatically
    log_hardware_summary(log_dir=log_dir)

    return log_file
