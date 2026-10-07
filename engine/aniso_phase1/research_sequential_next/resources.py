"""Conservative allocation guard plus explicitly sampled CUDA memory records."""
import resource
import warp as wp


class GPUMemoryBudget:
    def __init__(self,device,fraction=.7):
        self.device=wp.get_device(device)
        if not self.device.is_cuda or not 0<fraction<1:raise ValueError('CUDA device and fractional budget required')
        self.initial_free=int(self.device.free_memory);self.budget=int(fraction*self.initial_free)
        self.minimum_free=self.initial_free;self.peak_required=0;self.samples=0
        self.initial_pool_high=int(wp.get_mempool_used_mem_high(self.device))

    def observe(self,reserve_bytes=0):
        free=int(self.device.free_memory);self.minimum_free=min(self.minimum_free,free);self.samples+=1
        used=max(0,self.initial_free-free);required=used+int(reserve_bytes)
        self.peak_required=max(self.peak_required,required)
        if required>self.budget or reserve_bytes>free:raise MemoryError('resource_limited: CUDA allocation reserve exceeds frozen free-memory budget')
        return free

    def report(self):
        return dict(initial_free_bytes=self.initial_free,budget_bytes=self.budget,
            cuda_async_pool_current_bytes=int(wp.get_mempool_used_mem_current(self.device)),
            cuda_async_pool_process_high_bytes=int(wp.get_mempool_used_mem_high(self.device)),
            initial_cuda_async_pool_high_bytes=self.initial_pool_high,
            observed_device_free_decrease_bytes=max(0,self.initial_free-self.minimum_free),
            conservative_peak_required_bytes=self.peak_required,samples=self.samples,
            peak_process_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            scope='CUDA allocator reports exact process async-pool high water; free-memory samples additionally include non-pool/shared-device usage. Pool high water is process-wide, not per model; preallocation reserve is conservative.')
