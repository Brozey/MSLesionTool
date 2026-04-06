"""
convergence_monitor.py
======================
Hourly learning curve monitor that automatically detects training plateau and,
if enabled, stops the remote/local training early and restarts with the optimal
number of epochs — saving days of compute.

How it works:
    1. Every hour: parse nnU-Net training logs (remote via SSH, local from disk)
    2. Fit exponential saturation curve:  f(t) = a - b * exp(-c * t)
    3. Find the epoch where Dice reaches 99% of the projected plateau
    4. If that epoch saves ≥ MIN_EPOCH_SAVING epochs vs 1000:
       → Log the recommendation
       → If --auto-stop: gracefully kill training, patch run_all.sh with
         --num_epochs N, restart from checkpoint

Remote stop flow (nnU-Net handles SIGINT → saves checkpoint_latest.pth):
    kill -SIGINT <python_pid>          # graceful checkpoint save
    patch run_all.sh                   # add --num_epochs N to train calls
    screen -S training -X stuff "..."  # restart (resumes via --c)

Usage:
    # Monitor-only (no auto-stop), runs forever:
    python scripts/monitoring/convergence_monitor.py

    # Full auto — check every hour, stop early if plateau found:
    python scripts/monitoring/convergence_monitor.py --auto-stop

    # One-shot check right now:
    python scripts/monitoring/convergence_monitor.py --once

    # Tune sensitivity:
    python scripts/monitoring/convergence_monitor.py --auto-stop --threshold 0.99 --min-saving 100
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── optional matplotlib (only needed for --plot) ────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')          # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.optimize import curve_fit, OptimizeWarning
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

REMOTE_HOST        = "adafour@158.196.15.21"
REMOTE_RESULTS_DIR = "~/ms_lesion_seg/nnunet_workspace/nnUNet_results"
REMOTE_TRAIN_SCRIPT = "~/ms_lesion_seg/train_only.sh"     # patched with --num_epochs
REMOTE_LAUNCH       = "~/ms_lesion_seg/launch_training.sh"  # calls train_only.sh inside
REMOTE_SCREEN       = "training"           # screen session name pattern

LOCAL_RESULTS_DIR  = str(Path(os.environ.get("nnUNet_results", REPO_ROOT / "data" / "nnUNet_results")))
STATE_FILE         = str(REPO_ROOT / "convergence_state.json")
PLOTS_DIR          = str(REPO_ROOT / "plots" / "convergence")

DEFAULT_TOTAL_EPOCHS = 1000
DEFAULT_THRESHOLD    = 0.990   # 99.0% of plateau = convergence
DEFAULT_MIN_SAVING   = 150     # only act if we'd save ≥ 150 epochs
MIN_EPOCHS_TO_FIT    = 60      # need at least this many data points
MIN_R2               = 0.90    # curve fit quality gate
CHECK_INTERVAL_SEC   = 3600    # 1 hour

# ─────────────────────────────────────────────────────────────────────────────
# Experiment queue — local 2.5D experiments run one after another
# Remote chaining is handled by train_only.sh (already has all 4 in sequence)
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_EXPERIMENT_QUEUE = [
    {'ds_id': '500', 'plans': 'nnUNetPlans',            'tag': 'DS500 CNN'},
    {'ds_id': '501', 'plans': 'nnUNetPlans',            'tag': 'DS501 CNN'},
    {'ds_id': '500', 'plans': 'nnUNetResEncUNetLPlans', 'tag': 'DS500 ResEncL'},
    {'ds_id': '501', 'plans': 'nnUNetResEncUNetLPlans', 'tag': 'DS501 ResEncL'},
]

# Map dataset IDs to their directory names
DS_DIR_NAMES = {
    '500': 'Dataset500_RawFLAIR',
    '501': 'Dataset501_SkullStrippedFLAIR',
}


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_log(text: str) -> tuple[list[int], list[float]]:
    """Return (epochs, pseudo_dices) from a nnU-Net training log."""
    epochs, dices = [], []
    current_epoch = None
    for line in text.splitlines():
        m = re.search(r'Epoch (\d+)', line)
        if m:
            current_epoch = int(m.group(1))
        m = re.search(r'Pseudo dice \[.*?(\d+\.\d+)', line)
        if m and current_epoch is not None:
            epochs.append(current_epoch)
            dices.append(float(m.group(1)))
    return epochs, dices


def ssh(cmd: str, timeout: int = 60) -> str:
    """Run a command on the remote server, return stdout."""
    ssh_opts = (
        '-o ConnectTimeout=15 '
        '-o BatchMode=yes '
        '-o ServerAliveInterval=10 '
        '-o ServerAliveCountMax=3'
    )
    try:
        proc = subprocess.Popen(
            f'ssh {ssh_opts} {REMOTE_HOST} "{cmd}"',
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, _ = proc.communicate(timeout=timeout)
        return stdout
    except subprocess.TimeoutExpired:
        log(f"  SSH timeout ({timeout}s) for: {cmd[:80]}...")
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return ""
    except Exception as e:
        log(f"  SSH error: {type(e).__name__}: {e}")
        return ""


def fetch_remote_logs() -> dict[str, str]:
    """SSH fetch all training logs in a single connection. Returns {experiment_key: log_text}."""
    # Single SSH call: find all logs and cat them with delimiters
    script = (
        f"for f in $(find {REMOTE_RESULTS_DIR} -name 'training_log_*' | sort); do "
        f"echo '===LOGFILE===' $f; cat $f; done"
    )
    raw = ssh(script, timeout=120)
    if not raw.strip():
        return {}

    logs = {}
    current_key = None
    current_lines = []
    for line in raw.splitlines():
        if line.startswith('===LOGFILE=== '):
            # Save previous log
            if current_key and current_lines:
                logs[current_key] = '\n'.join(current_lines)
            # Parse new path
            path = line[len('===LOGFILE=== '):]
            parts = path.split('/')
            try:
                dataset = next(p for p in parts if p.startswith('Dataset'))
                trainer = next(p for p in parts if '__' in p and 'nnUNet' in p)
                current_key = f"[REMOTE] {dataset}/{trainer}"
            except StopIteration:
                current_key = f"[REMOTE] {path}"
            current_lines = []
        else:
            current_lines.append(line)
    # Save last log
    if current_key and current_lines:
        logs[current_key] = '\n'.join(current_lines)
    return logs


def find_local_logs() -> dict[str, str]:
    """Find all local training logs."""
    root = Path(LOCAL_RESULTS_DIR)
    logs = {}
    for p in sorted(root.rglob("training_log_*.txt")):
        parts = p.parts
        try:
            dataset = next(x for x in parts if x.startswith('Dataset'))
            trainer = next(x for x in parts if '__' in x and 'nnUNet' in x)
            key = f"[LOCAL] {dataset}/{trainer}"
        except StopIteration:
            key = f"[LOCAL] {p.name}"
        # Merge duplicate keys (multiple log files = restarts → take newest)
        if key not in logs or p.stat().st_mtime > Path(LOCAL_RESULTS_DIR).stat().st_mtime:
            logs[key] = p.read_text(encoding='utf-8', errors='ignore')
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Curve fitting
# ─────────────────────────────────────────────────────────────────────────────

def exp_saturation(t, a, b, c):
    return a - b * np.exp(-c * t)


def r_squared(y_true, y_pred) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_convergence(epochs: list[int], dices: list[float]
                    ) -> dict | None:
    """
    Fit exp saturation curve. Returns result dict or None if insufficient data.
    Result keys: plateau, b, c, r2, epoch_99, epoch_995, epoch_999

    Applies EMA smoothing to reduce epoch-to-epoch noise (lesion dice is noisy)
    before fitting, which dramatically improves R².
    """
    if not HAS_SCIPY or len(epochs) < MIN_EPOCHS_TO_FIT:
        return None

    t = np.array(epochs, dtype=float)
    y = np.array(dices, dtype=float)

    # EMA smoothing — reduces noise while preserving the trend shape
    # alpha=0.05 → effective window ~20 epochs
    alpha = 0.05
    y_smooth = np.empty_like(y)
    y_smooth[0] = y[0]
    for i in range(1, len(y)):
        y_smooth[i] = alpha * y[i] + (1 - alpha) * y_smooth[i - 1]

    a0 = min(max(y_smooth) * 1.10, 0.99)
    b0 = max(a0 - y_smooth[0], 0.01)
    c0 = 0.005

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                exp_saturation, t, y_smooth,
                p0=[a0, b0, c0],
                bounds=([0.01, 0.0, 1e-6], [1.0, 2.0, 1.0]),
                maxfev=20000,
            )
    except Exception:
        return None

    a, b, c = popt
    y_pred = exp_saturation(t, a, b, c)
    r2 = r_squared(y_smooth, y_pred)

    if r2 < MIN_R2:
        return None

    def epoch_at_threshold(thresh: float) -> int:
        target = thresh * a
        ratio = (a - target) / max(b, 1e-9)
        if ratio <= 0 or ratio >= 1:
            return DEFAULT_TOTAL_EPOCHS
        t_val = -np.log(ratio) / c
        return max(1, int(np.ceil(t_val)))

    return {
        'plateau':    round(float(a), 4),
        'b':          round(float(b), 4),
        'c':          round(float(c), 6),
        'r2':         round(r2, 4),
        'epoch_990':  epoch_at_threshold(0.990),
        'epoch_995':  epoch_at_threshold(0.995),
        'epoch_999':  epoch_at_threshold(0.999),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        state = json.loads(Path(STATE_FILE).read_text())
    except Exception:
        state = {}
    # Ensure all expected keys exist
    state.setdefault('stopped', {})
    state.setdefault('last_check', None)
    state.setdefault('analyses', {})
    state.setdefault('chain_completed', [])
    state.setdefault('chain_log', [])
    return state


def save_state(state: dict):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Remote control
# ─────────────────────────────────────────────────────────────────────────────

def get_remote_training_pid() -> int | None:
    """Find PID of the running nnUNetv2_train python process on remote."""
    out = ssh("pgrep -f 'nnUNetv2_train' | head -1")
    pid = out.strip()
    return int(pid) if pid.isdigit() else None


def kill_remote_training() -> bool:
    """
    Send SIGINT to remote nnU-Net process.
    nnU-Net catches KeyboardInterrupt and saves checkpoint_latest.pth before exit.
    Returns True if a process was found and signalled.
    """
    pid = get_remote_training_pid()
    if pid is None:
        log("  No active training process found on remote.")
        return False
    log(f"  Sending SIGINT to PID {pid} (will save checkpoint then exit)...")
    ssh(f"kill -SIGINT {pid}")
    # Wait up to 120s for the process to finish saving
    for _ in range(24):
        time.sleep(5)
        remaining = ssh(f"ps -p {pid} -o pid= 2>/dev/null").strip()
        if not remaining:
            log("  Process exited cleanly.")
            return True
    log("  WARNING: process still running after 120s — manual intervention may be needed.")
    return True


def _extract_remote_fold_dir(key: str) -> str | None:
    """
    Extract the remote fold_0 directory path from an experiment key.
    Key format: '[REMOTE] DatasetNNN_Name/nnUNetTrainer__Plans__3d_fullres'
    """
    m = re.search(r'\[REMOTE\]\s+(Dataset\S+)/(nnUNet\S+)', key)
    if not m:
        return None
    ds_dir, trainer_dir = m.group(1), m.group(2)
    return f"{REMOTE_RESULTS_DIR}/{ds_dir}/{trainer_dir}/fold_0"


def mark_remote_experiment_done(key: str) -> bool:
    """
    Mark a remote experiment as done by copying checkpoint_latest.pth
    to checkpoint_final.pth.  train_only.sh checks for checkpoint_final
    and will skip completed experiments automatically.
    """
    fold_dir = _extract_remote_fold_dir(key)
    if fold_dir is None:
        log(f"  ERROR: could not parse fold dir from key: {key}")
        return False

    latest = f"{fold_dir}/checkpoint_latest.pth"
    final  = f"{fold_dir}/checkpoint_final.pth"

    # Verify checkpoint_latest exists
    try:
        check = ssh(f"test -f {latest} && echo EXISTS", timeout=30).strip()
    except Exception as e:
        log(f"  ERROR checking checkpoint_latest: {type(e).__name__}: {e}")
        return False
    if check != 'EXISTS':
        log(f"  ERROR: {latest} not found on remote")
        return False

    try:
        ssh(f"cp {latest} {final}", timeout=120)
        verify = ssh(f"test -f {final} && echo OK", timeout=30).strip()
    except Exception as e:
        log(f"  ERROR copying checkpoint: {type(e).__name__}: {e}")
        return False
    if verify == 'OK':
        log(f"  checkpoint_final.pth created (copied from checkpoint_latest.pth)")
        return True
    else:
        log(f"  ERROR: failed to create checkpoint_final.pth")
        return False


def restart_remote_training():
    """Resume training in the existing screen session."""
    screen_name = ssh("screen -ls | grep training | awk '{print $1}' | head -1").strip()
    if not screen_name:
        log("  WARNING: no 'training' screen session found. Creating new one...")
        ssh(f"screen -dmS training bash {REMOTE_LAUNCH}")
    else:
        log(f"  Sending restart command to screen session: {screen_name}")
        ssh(f"screen -S {screen_name} -X stuff 'bash {REMOTE_LAUNCH}\\n'")
    log("  Remote training restarted (will auto-resume from checkpoint).")


# ─────────────────────────────────────────────────────────────────────────────
# Local control (Windows)
# ─────────────────────────────────────────────────────────────────────────────

def get_local_training_pid() -> int | None:
    """Find PID of the local python process running nnUNetv2_train."""
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\""
             " | Where-Object {$_.CommandLine -like '*nnUNetv2_train*'}"
             " | Select-Object -First 1).ProcessId"],
            capture_output=True, text=True, timeout=15
        )
        pid = result.stdout.strip()
        return int(pid) if pid.isdigit() else None
    except Exception:
        return None


def kill_local_training() -> bool:
    """
    Force-kill the local nnUNetv2_train python process.

    On Windows we cannot send SIGINT across console sessions, so we use
    taskkill.  nnU-Net saves checkpoint_latest.pth at the end of every
    epoch, so we lose at most the in-progress epoch — which is fine
    because we immediately restart with --c (resume from checkpoint).
    """
    pid = get_local_training_pid()
    if pid is None:
        log("  No active local training process found.")
        return False
    log(f"  Killing local training PID {pid} (checkpoint_latest.pth already saved after last epoch)...")
    subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                   capture_output=True, timeout=15)
    time.sleep(3)
    # Verify it's gone
    if get_local_training_pid() is None:
        log("  Process terminated.")
        return True
    log("  WARNING: process may still be running.")
    return True


def restart_local_training(key: str, num_epochs: int):
    """
    Launch a new PowerShell window with the nnUNetv2_train command.
    Resumes from checkpoint with --c and --num_epochs N.
    """
    ds_match = re.search(r'Dataset(\d+)', key)
    ds_id = ds_match.group(1) if ds_match else "500"
    is_resencl = 'ResEncL' in key or 'ResEncUNetL' in key

    plans = '-p nnUNetResEncUNetLPlans' if is_resencl else ''
    train_cmd = (
        f'nnUNetv2_train {ds_id} 3d_fullres 0 -tr nnUNetTrainer_25D '
        f'{plans} --num_epochs {num_epochs} --c'
    ).replace('  ', ' ')

    _launch_local_powershell(train_cmd)
    log(f"  Local training restarted in new PowerShell window: {train_cmd}")


def _launch_local_powershell(train_cmd: str):
    """Launch a PowerShell window with nnUNet env vars and the given command."""
    _raw = os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw"))
    _pre = os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "data" / "nnUNet_preprocessed"))
    _res = os.environ.get("nnUNet_results", str(REPO_ROOT / "data" / "nnUNet_results"))
    ps_script = (
        f'$env:nnUNet_raw = \\"{_raw}\\"; '
        f'$env:nnUNet_preprocessed = \\"{_pre}\\"; '
        f'$env:nnUNet_results = \\"{_res}\\"; '
        f'$env:KMP_DUPLICATE_LIB_OK = \\"TRUE\\"; '
        f'cd {REPO_ROOT}; '
        f'{train_cmd}'
    )
    subprocess.Popen(
        f'start "nnUNet_auto" powershell -NoExit -Command "{ps_script}"',
        shell=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment chaining — detect completion, start next experiment
# ─────────────────────────────────────────────────────────────────────────────

def _local_result_dir(exp: dict) -> Path:
    """Return the fold_0 result directory for a local experiment."""
    ds_dir = DS_DIR_NAMES[exp['ds_id']]
    trainer_dir = f"nnUNetTrainer_25D__{exp['plans']}__3d_fullres"
    return Path(LOCAL_RESULTS_DIR) / ds_dir / trainer_dir / 'fold_0'


def _is_experiment_complete(exp: dict) -> bool:
    """Check if checkpoint_final.pth exists (training finished all epochs)."""
    result_dir = _local_result_dir(exp)
    return (result_dir / 'checkpoint_final.pth').exists()


def _is_experiment_started(exp: dict) -> bool:
    """Check if training has started (any checkpoint or log exists)."""
    result_dir = _local_result_dir(exp)
    if not result_dir.exists():
        return False
    return any(result_dir.glob('checkpoint_*.pth')) or any(result_dir.glob('training_log_*.txt'))


def _ensure_preprocessed(exp: dict) -> bool:
    """
    Ensure preprocessed data exists for the experiment.
    ResEncL plans may share the same data_identifier as the default plans,
    so we read the plans JSON to find the actual preprocessed folder name.
    Returns True if ready, False if preprocessing failed.
    """
    preproc_dir = Path(os.environ.get("nnUNet_preprocessed", REPO_ROOT / "data" / "nnUNet_preprocessed"))
    ds_dir = DS_DIR_NAMES[exp['ds_id']]

    # Read the plans JSON to get the actual data_identifier
    plans_json = preproc_dir / ds_dir / f"{exp['plans']}.json"
    if plans_json.exists():
        import json
        plans = json.loads(plans_json.read_text())
        data_id = plans.get('configurations', {}).get('3d_fullres', {}).get(
            'data_identifier', f"{exp['plans']}_3d_fullres")
    else:
        data_id = f"{exp['plans']}_3d_fullres"

    expected = preproc_dir / ds_dir / data_id

    if expected.exists() and any(expected.iterdir()):
        log(f"  Preprocessed data found for {exp['tag']} ({data_id})")
        return True

    log(f"  Preprocessed data missing for {exp['tag']} ({data_id})")
    log(f"  Running nnUNetv2_preprocess -d {exp['ds_id']} -plans_name {exp['plans']} -c 3d_fullres ...")

    try:
        _raw = os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw"))
        _pre = os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "data" / "nnUNet_preprocessed"))
        _res = os.environ.get("nnUNet_results", str(REPO_ROOT / "data" / "nnUNet_results"))
        result = subprocess.run(
            ['powershell', '-Command',
             f'$env:nnUNet_raw = "{_raw}"; '
             f'$env:nnUNet_preprocessed = "{_pre}"; '
             f'$env:nnUNet_results = "{_res}"; '
             f'$env:KMP_DUPLICATE_LIB_OK = "TRUE"; '
             f'nnUNetv2_preprocess -d {exp["ds_id"]} -plans_name {exp["plans"]} -c 3d_fullres -np 1'],
            capture_output=True, text=True, timeout=7200  # 2h max
        )
        if expected.exists() and any(expected.iterdir()):
            log(f"  Preprocessing complete for {exp['tag']}.")
            return True
        else:
            log(f"  WARNING: preprocessing ran but output not found.")
            log(f"  stderr: {result.stderr[-500:] if result.stderr else '(none)'}")
            return False
    except Exception as e:
        log(f"  ERROR during preprocessing: {e}")
        return False


def check_and_chain_local_experiments(state: dict) -> dict:
    """
    Check if the current local experiment is done.
    If so, start the next one from the queue.
    """
    training_pid = get_local_training_pid()

    # Show queue status
    log("\nLocal experiment queue:")
    current_idx = None
    for i, exp in enumerate(LOCAL_EXPERIMENT_QUEUE):
        complete = _is_experiment_complete(exp)
        started = _is_experiment_started(exp)
        early_stopped = exp['tag'] in state.get('chain_completed', [])
        if complete or early_stopped:
            status = "DONE" if complete else "DONE (early-stopped)"
        elif started:
            status = "RUNNING" if training_pid else "STOPPED (incomplete)"
            if current_idx is None:
                current_idx = i
        else:
            status = "queued"
        log(f"  {i+1}. {exp['tag']:15s} [{status}]")

    # If training is actively running, nothing to chain
    if training_pid:
        log(f"  Training process active (PID {training_pid}) — no chaining needed.")
        return state

    # Find the next experiment to run
    next_exp = None
    next_idx = None
    for i, exp in enumerate(LOCAL_EXPERIMENT_QUEUE):
        if not _is_experiment_complete(exp) and exp['tag'] not in state.get('chain_completed', []):
            next_exp = exp
            next_idx = i
            break

    if next_exp is None:
        log("  All local experiments COMPLETE!")
        return state

    # Check if this experiment was already started but training stopped
    if _is_experiment_started(next_exp):
        # Has a checkpoint but no final — check if early-stopped by us
        result_dir = _local_result_dir(next_exp)
        key_pattern = f"[LOCAL] {DS_DIR_NAMES[next_exp['ds_id']]}/nnUNetTrainer_25D__{next_exp['plans']}__3d_fullres"
        if key_pattern in state.get('stopped', {}):
            log(f"  Experiment {next_exp['tag']} was early-stopped. Moving to next.")
            state.setdefault('chain_completed', []).append(next_exp['tag'])
            # Find truly next experiment
            found_next = False
            for i2, exp2 in enumerate(LOCAL_EXPERIMENT_QUEUE[next_idx+1:], next_idx+1):
                if not _is_experiment_complete(exp2) and exp2['tag'] not in state.get('chain_completed', []):
                    next_exp = exp2
                    next_idx = i2
                    found_next = True
                    break
            if not found_next:
                log("  All local experiments done (some early-stopped).")
                return state
        else:
            # Training crashed or was manually stopped — resume it
            log(f"\n  Experiment {next_exp['tag']} has checkpoint but no running process.")
            log(f"  Resuming from checkpoint...")
            plans_flag = f'-p {next_exp["plans"]}' if next_exp['plans'] != 'nnUNetPlans' else ''
            train_cmd = (
                f'nnUNetv2_train {next_exp["ds_id"]} 3d_fullres 0 '
                f'-tr nnUNetTrainer_25D {plans_flag} --c'
            ).replace('  ', ' ')
            _launch_local_powershell(train_cmd)
            log(f"  Resumed: {train_cmd}")
            state.setdefault('chain_log', []).append({
                'ts': datetime.now().isoformat(),
                'action': 'resume',
                'experiment': next_exp['tag'],
            })
            return state

    # Fresh start for next experiment
    log(f"\n★ CHAINING: Starting next experiment — {next_exp['tag']}")

    # Ensure preprocessed data exists (especially for ResEncL)
    if not _ensure_preprocessed(next_exp):
        log(f"  Cannot start {next_exp['tag']} — preprocessing failed.")
        return state

    plans_flag = f'-p {next_exp["plans"]}' if next_exp['plans'] != 'nnUNetPlans' else ''
    train_cmd = (
        f'nnUNetv2_train {next_exp["ds_id"]} 3d_fullres 0 '
        f'-tr nnUNetTrainer_25D {plans_flag}'
    ).replace('  ', ' ')

    _launch_local_powershell(train_cmd)
    log(f"  Started: {train_cmd}")

    state.setdefault('chain_log', []).append({
        'ts': datetime.now().isoformat(),
        'action': 'start',
        'experiment': next_exp['tag'],
    })

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(key: str, epochs: list[int], dices: list[float],
              fit: dict | None, optimal_epoch: int | None):
    if not HAS_MPL:
        return
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    t = np.array(epochs)
    y = np.array(dices)
    ax.plot(t, y, 'b.', alpha=0.4, markersize=3, label='Pseudo Dice / epoch')

    # Smoothed
    window = min(15, max(3, len(y) // 10))
    if len(y) >= window * 2:
        smoothed = np.convolve(y, np.ones(window) / window, mode='valid')
        ax.plot(t[window - 1:], smoothed, 'b-', lw=1.5, label=f'EMA (w={window})')

    if fit:
        t_full = np.linspace(0, DEFAULT_TOTAL_EPOCHS, 600)
        y_fit = exp_saturation(t_full, fit['plateau'], fit['b'], fit['c'])
        ax.plot(t_full, y_fit, 'r--', lw=1.5,
                label=f"fit: plateau={fit['plateau']:.3f}, R²={fit['r2']:.3f}")
        ax.axhline(fit['plateau'], color='red', ls='--', alpha=0.25, lw=0.8)

        for ep, thresh, color in [
            (fit['epoch_990'],  '99.0%', 'orange'),
            (fit['epoch_995'],  '99.5%', 'darkorange'),
        ]:
            if ep <= DEFAULT_TOTAL_EPOCHS:
                ax.axvline(ep, color=color, ls=':', alpha=0.8,
                           label=f'{thresh} plateau @ ep{ep}')

    if optimal_epoch:
        ax.axvline(optimal_epoch, color='green', ls='-', lw=2, alpha=0.7,
                   label=f'STOP @ ep{optimal_epoch}')

    ax.set_title(key, fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pseudo Dice')
    ax.set_xlim(0, DEFAULT_TOTAL_EPOCHS)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')
    ax.text(0.01, 0.97, f"As of {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            transform=ax.transAxes, fontsize=7, va='top', color='gray')

    safe_key = re.sub(r'[^\w\-]', '_', key)[:60]
    out_path = Path(PLOTS_DIR) / f"{safe_key}.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    log(f"  Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Replace Unicode chars that cp1250/cp1252 can't encode
    safe = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis engine
# ─────────────────────────────────────────────────────────────────────────────

def analyse_all(auto_stop: bool, threshold: float, min_saving: int,
                state: dict) -> dict:
    """
    Analyse all logs, printing results. Optionally trigger early stopping.
    Returns updated state.
    """
    all_logs = {}
    log("Fetching remote logs...")
    try:
        all_logs.update(fetch_remote_logs())
    except Exception as e:
        log(f"  Remote fetch failed: {e}")

    log("Scanning local logs...")
    all_logs.update(find_local_logs())

    if not all_logs:
        log("No training logs found.")
        return state

    for key, text in sorted(all_logs.items()):
        epochs, dices = parse_log(text)
        if not epochs:
            continue

        current_ep = epochs[-1]
        best_dice  = max(dices)
        log(f"\n{'─'*60}")
        log(f"Experiment : {key}")
        log(f"Progress   : epoch {current_ep}/{DEFAULT_TOTAL_EPOCHS} "
            f"({current_ep/DEFAULT_TOTAL_EPOCHS*100:.0f}%)")
        log(f"Best Dice  : {best_dice:.4f}  |  Latest: {dices[-1]:.4f}")

        fit = fit_convergence(epochs, dices)

        if fit is None:
            if len(epochs) < MIN_EPOCHS_TO_FIT:
                log(f"Fit        : need ≥{MIN_EPOCHS_TO_FIT} epochs "
                    f"(have {len(epochs)}) — too early")
            else:
                log(f"Fit        : curve did not converge or R²<{MIN_R2}")
        else:
            optimal = fit[f'epoch_{int(threshold*1000):d}']
            # Clamp to feasible range
            optimal = max(optimal, current_ep + 1)
            saving  = DEFAULT_TOTAL_EPOCHS - optimal

            log(f"Fit        : plateau={fit['plateau']:.4f}, R²={fit['r2']:.3f}")
            log(f"Convergence: 99.0%→ep{fit['epoch_990']}  "
                f"99.5%→ep{fit['epoch_995']}  99.9%→ep{fit['epoch_999']}")
            log(f"Optimal    : stop at epoch {optimal} "
                f"(saves {saving} epochs = "
                f"{saving*208/3600:.1f}h remote / {saving*120/3600:.1f}h local)")

            # Save to state
            state['analyses'][key] = {
                'ts': datetime.now().isoformat(),
                'current_epoch': current_ep,
                'optimal_epoch': optimal,
                'plateau': fit['plateau'],
                'r2': fit['r2'],
                'saving': saving,
            }

            save_plot(key, epochs, dices, fit, optimal)

            # ── Auto-stop logic ──────────────────────────────────────
            if auto_stop:
                already_stopped = key in state['stopped']
                already_past    = current_ep >= optimal

                if already_stopped:
                    log(f"Auto-stop  : already triggered for this experiment — skipping")
                elif already_past:
                    log(f"Auto-stop  : already past optimal epoch — no action")
                elif saving < min_saving:
                    log(f"Auto-stop  : saving ({saving}) < min_saving ({min_saving}) — not worth it")
                elif fit['r2'] < 0.95:
                    log(f"Auto-stop  : fit R²={fit['r2']:.3f} < 0.95 — not confident enough yet")
                elif current_ep < MIN_EPOCHS_TO_FIT * 2:
                    log(f"Auto-stop  : only {current_ep} epochs — need more data before acting")
                else:
                    # ✓ All checks passed — trigger early stop
                    if '[REMOTE]' in key:
                        log(f"\n* AUTO-STOP triggered for {key}")
                        log(f"  Target: stop at epoch {optimal} (saving {saving} epochs)")
                        try:
                            killed = kill_remote_training()
                            if killed:
                                time.sleep(10)  # let checkpoint save
                                marked = mark_remote_experiment_done(key)
                                if marked:
                                    time.sleep(5)
                                    restart_remote_training()
                                    state['stopped'][key] = {
                                        'ts': datetime.now().isoformat(),
                                        'optimal_epoch': optimal,
                                        'saving': saving,
                                    }
                                    log(f"  Done. Marked complete, restarted training.")
                                else:
                                    log("  ERROR: could not mark experiment as done.")
                                    log(f"  Manual fix: cp checkpoint_latest.pth checkpoint_final.pth, then restart screen")
                        except Exception as e:
                            log(f"  EXCEPTION during remote auto-stop: {type(e).__name__}: {e}")
                            log(f"  Manual fix: cp checkpoint_latest.pth checkpoint_final.pth, then restart screen")
                    else:
                        # ── Local auto-stop ──
                        log(f"\n★ AUTO-STOP for {key}:")
                        try:
                            killed = kill_local_training()
                            if killed:
                                time.sleep(5)  # let OS release resources
                                restart_local_training(key, optimal)
                                state['stopped'][key] = {
                                    'ts': datetime.now().isoformat(),
                                    'optimal_epoch': optimal,
                                    'saving': saving,
                                }
                                log(f"  ✓ Done. Restarted with --num_epochs {optimal}")
                            else:
                                log("  Could not find running training process.")
                                log("  → Stop it manually, then restart with:")
                                local_cmds = _build_local_restart_commands(key, optimal)
                                for cmd in local_cmds:
                                    log(f"    {cmd}")
                        except Exception as e:
                            log(f"  EXCEPTION during local auto-stop: {type(e).__name__}: {e}")

    return state


def _build_local_restart_commands(key: str, num_epochs: int) -> list[str]:
    """Build the PowerShell commands to restart local training with fewer epochs."""
    # Parse dataset ID from key
    ds_match = re.search(r'Dataset(\d+)', key)
    ds_id = ds_match.group(1) if ds_match else "500"
    is_resencl = 'ResEncL' in key or 'ResEncUNetL' in key

    env = (
        f'$env:nnUNet_raw = "{os.environ.get("nnUNet_raw", str(REPO_ROOT / "data" / "nnUNet_raw"))}"; '
        f'$env:nnUNet_preprocessed = "{os.environ.get("nnUNet_preprocessed", str(REPO_ROOT / "data" / "nnUNet_preprocessed"))}"; '
        f'$env:nnUNet_results = "{os.environ.get("nnUNet_results", str(REPO_ROOT / "data" / "nnUNet_results"))}"; '
        f'$env:KMP_DUPLICATE_LIB_OK = "TRUE"'
    )
    train_cmd = f'nnUNetv2_train {ds_id} 3d_fullres 0 -tr nnUNetTrainer_25D --num_epochs {num_epochs} --c'
    if is_resencl:
        train_cmd = f'nnUNetv2_train {ds_id} 3d_fullres 0 -tr nnUNetTrainer_25D -p nnUNetResEncUNetLPlans --num_epochs {num_epochs} --c'

    return [env, train_cmd]


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="nnU-Net convergence monitor")
    parser.add_argument('--auto-stop', action='store_true',
                        help='Automatically stop and restart training when plateau detected')
    parser.add_argument('--once', action='store_true',
                        help='Run one check and exit (default: loop every hour)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Plateau threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--min-saving', type=int, default=DEFAULT_MIN_SAVING,
                        help=f'Only act if saving ≥ N epochs (default: {DEFAULT_MIN_SAVING})')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL_SEC,
                        help=f'Check interval in seconds (default: {CHECK_INTERVAL_SEC})')
    args = parser.parse_args()

    if not HAS_SCIPY:
        log("ERROR: scipy not installed. Run: pip install scipy")
        sys.exit(1)

    log("=" * 60)
    log("  nnU-Net Convergence Monitor + Experiment Chainer")
    log(f"  Auto-stop : {'ENABLED' if args.auto_stop else 'disabled (monitor only)'}")
    log(f"  Chaining  : ENABLED (4 local experiments in queue)")
    log(f"  Threshold : {args.threshold*100:.1f}% of plateau")
    log(f"  Min saving: {args.min_saving} epochs")
    log(f"  Interval  : {args.interval}s")
    log("=" * 60)

    state = load_state()

    while True:
        log(f"\n{'='*60}")
        log(f"CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'='*60}")

        state = analyse_all(
            auto_stop=args.auto_stop,
            threshold=args.threshold,
            min_saving=args.min_saving,
            state=state,
        )

        # ── Experiment chaining: if current experiment finished, start next ──
        state = check_and_chain_local_experiments(state)

        state['last_check'] = datetime.now().isoformat()
        save_state(state)

        if args.once:
            log("\nDone (--once mode).")
            break

        next_check = datetime.fromtimestamp(time.time() + args.interval)
        log(f"\nNext check: {next_check.strftime('%H:%M:%S')} "
            f"(in {args.interval//60} min). Ctrl+C to stop.")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
