"""Record the exact free-memory sample used by the inherited allocation guard.

The proxy intercepts one property read; it does not repeat/reimplement the
guard's decision. Reports on a rejected allocation are durable immediately.
"""
from pathlib import Path
import argparse
import os
import runpy
import subprocess
import sys
from .provenance import write, utc, sha


class SampledDevice:
    def __init__(self, device):
        self.device = device
        self.sample = None

    @property
    def free_memory(self):
        self.sample = int(self.device.free_memory)
        return self.sample

    def __getattr__(self, name):
        return getattr(self.device, name)


def sample_guard(guard, original, reserve_bytes=0):
    device = guard.device
    proxy = SampledDevice(device)
    guard.device = proxy
    try:
        value = original(guard, reserve_bytes)
        return value, proxy.sample, None
    except MemoryError as error:
        return None, proxy.sample, error
    finally:
        guard.device = device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('module')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    import warp as wp
    from engine.aniso_phase1.research_sequential_next.resources import GPUMemoryBudget
    original = GPUMemoryBudget.observe
    records, failures = [], []
    status = 'failed'

    def report():
        write(a.report, dict(status=status, pid=os.getpid(), samples=records,
                            failures=failures, guard_unchanged=True,
                            phase=dict(module=a.module,arguments=a.arguments),
                            observer_sha256=sha(__file__)))

    def observe(guard, reserve_bytes=0):
        nonlocal status
        value, free, error = sample_guard(guard, original, reserve_bytes)
        used = max(0, guard.initial_free - free)
        row = dict(utc=utc(), initial_free=guard.initial_free,
                   decision_free_bytes=free, budget=guard.budget,
                   reserve_bytes=int(reserve_bytes), required_bytes=used+int(reserve_bytes),
                   required_over_budget=used+int(reserve_bytes)>guard.budget,
                   reserve_over_free=reserve_bytes>free,
                   pool_current=int(wp.get_mempool_used_mem_current(guard.device)),
                   pool_process_high=int(wp.get_mempool_used_mem_high(guard.device)))
        records.append(row)
        if error is not None:
            status = 'resource_limited'
            failures.append(dict(**row, error=str(error)))
            report()  # Persist the decision before supplementary sampling.
            try:
                failures[-1]['supplementary_nvidia_smi'] = subprocess.run(
                    ['nvidia-smi'], capture_output=True, text=True, timeout=5).stdout
            finally:
                report()
            raise error
        return value

    GPUMemoryBudget.observe = observe
    sys.argv = [a.module, *a.arguments]
    try:
        runpy.run_module(a.module, run_name='__main__')
        status = 'completed'
    finally:
        GPUMemoryBudget.observe = original
        report()


if __name__ == '__main__':
    main()
