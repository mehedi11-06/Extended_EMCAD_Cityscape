"""GPU memory and step-time profiling.

Step time definition used throughout this project (state it in the README):

    step time = forward + loss + backward + optimizer.step + zero_grad,
                measured on the GPU with CUDA synchronization on both sides,
                EXCLUDING dataloading and EXCLUDING validation.

Dataloading is excluded because it is CPU/IO-bound and depends on worker count
and disk, which would make the number say more about the machine than about
the model. A separate `data_wait_s` is logged so the exclusion is visible
rather than hidden.

CUDA kernel launches are asynchronous: without torch.cuda.synchronize() a
CPU-side timer measures launch overhead, not compute.
"""

import json
import os
import subprocess
import time

import torch


def env_info(extra=None):
    """Hardware / software facts that must accompany every efficiency number."""
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            if torch.cuda.is_available() else None),
        "driver": _nvidia_driver(),
    }
    if extra:
        info.update(extra)
    return info


def _nvidia_driver():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb():
    """Returns (allocated, reserved) in GiB.

    allocated = tensor memory actually held by PyTorch -> report this one.
    reserved  = what the caching allocator took from the driver (always >=).
    nvidia-smi shows reserved plus the CUDA context, which is why it is the
    wrong instrument for a model-level claim.
    """
    if not torch.cuda.is_available():
        return (float("nan"), float("nan"))
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() / 1024 ** 3,
            torch.cuda.max_memory_reserved() / 1024 ** 3)


class StepTimer:
    """Times GPU work with synchronization, discarding warm-up iterations.

        timer = StepTimer(warmup=10)
        for ...:
            timer.start()
            ... forward/loss/backward/step ...
            timer.stop()
        timer.mean(), timer.std(), timer.count
    """

    def __init__(self, warmup=10):
        self.warmup = warmup
        self.times = []
        self.all_times = []          # kept so very short runs still report something
        self.seen = 0
        self._t0 = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.seen += 1
        self.all_times.append(dt)
        if self.seen > self.warmup:          # exclude warm-up from the average
            self.times.append(dt)
        return dt

    @property
    def count(self):
        return len(self.times)

    def _sample(self):
        # If the run was shorter than the warm-up, fall back to every timing
        # and flag it, rather than reporting NaN.
        return (self.times, False) if self.times else (self.all_times, True)

    def mean(self):
        import statistics
        s, _ = self._sample()
        return statistics.fmean(s) if s else float("nan")

    def std(self):
        import statistics
        s, _ = self._sample()
        return statistics.pstdev(s) if len(s) > 1 else 0.0

    def report(self):
        s, fallback = self._sample()
        return {"mean_step_time_s": self.mean(),
                "std_step_time_s": self.std(),
                "timed_iterations": len(s),
                "warmup_iterations": 0 if fallback else self.warmup,
                "warmup_excluded": not fallback}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    return path
