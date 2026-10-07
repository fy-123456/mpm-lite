"""Explicit inherited package loader; absence is never an original144 fallback."""
from benchmarks.research_spatial_phase_next.spaces import load_selected as inherited_load

def load_selected(entry):
    if not isinstance(entry,dict) or not entry.get('path') or not entry.get('sha256'):
        raise ValueError('explicit qualified physical space required')
    return inherited_load(entry)
