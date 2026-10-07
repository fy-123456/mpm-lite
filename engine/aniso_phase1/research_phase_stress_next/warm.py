"""Optional coarse instrumentation; numerical launches and allocation guards unchanged."""
import time
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedGPUOperator
from engine.aniso_phase1.research_d.identity import digest


class WarmOperator(SegmentedGPUOperator):
    def _stamp(self,name,started):
        # Device-to-host responses already wait for required data. Intermediate
        # diagnostic barriers are unnecessary for correct stream ordering.
        # These counters are host elapsed times, deliberately not GPU partitions.
        now=time.perf_counter();self.timings[name]+=now-started;self.counts[name]+=1
        if name in ('host_stabilization_and_identity','exact_tangent_total'):
            self.memory_budget.observe()
        return time.perf_counter()


def install_warm(model):
    if not isinstance(model.operator,SegmentedGPUOperator):raise ValueError('segmented operator required')
    model.operator.__class__=WarmOperator
    model.operator.signature=digest(dict(parent=model.operator.signature,instrumentation='coarse-host-no-intermediate-barriers-v1'))
    model._cached_q=None;model._cached_response=None
    return model
