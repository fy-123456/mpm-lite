"""Read-only resource sampling around an unchanged numerical driver."""
from pathlib import Path
import argparse,runpy,sys,subprocess
from .provenance import write,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('module');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    import warp as wp
    from engine.aniso_phase1.research_sequential_next.resources import GPUMemoryBudget
    original=GPUMemoryBudget.observe;records=[];failures=[]
    def observe(self,reserve_bytes=0):
        try:return original(self,reserve_bytes)
        except MemoryError:
            failures.append(dict(utc=utc(),reserve_bytes=int(reserve_bytes),budget=self.report(),live_device_free=int(self.device.free_memory),device_report=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10).stdout))
            write(a.report,dict(status='resource_limited',samples=records,failures=failures,guard_unchanged=True))
            raise
        finally:
            records.append(dict(initial_free=self.initial_free,budget=self.budget,min_observed_free=self.minimum_free,peak_required=self.peak_required,reserve_bytes=int(reserve_bytes),pool_current=int(wp.get_mempool_used_mem_current(self.device))))
    GPUMemoryBudget.observe=observe;sys.argv=[a.module,*a.arguments];status='failed'
    try:runpy.run_module(a.module,run_name='__main__');status='completed'
    finally:write(a.report,dict(status=status,samples=records,failures=failures,guard_unchanged=True))


if __name__=='__main__':main()
