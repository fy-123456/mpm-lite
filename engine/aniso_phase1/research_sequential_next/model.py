"""Read-only response values and explicit identity on the original M5 model."""
import numpy as np
from ..research_sequential.condensation import CondensedModel


def readonly_response(response):
    result={}
    for key,value in response.items():
        if isinstance(value,np.ndarray):
            value=np.array(value,copy=True);value.setflags(write=False)
        result[key]=value
    return result


class PracticalModel(CondensedModel):
    def evaluate(self,q,direction=None):
        # A caller must not mutate the inherited one-state cache through arrays
        # returned by its shallow dictionary copy. N08 will add bounded reuse.
        return readonly_response(super().evaluate(q,direction))
