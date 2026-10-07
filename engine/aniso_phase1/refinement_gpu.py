"""Optional float64 constitutive evaluation for the CPU reference equations."""
import numpy as np
import warp as wp


@wp.kernel
def response_kernel(Fs: wp.array(dtype=wp.mat33d), As: wp.array(dtype=wp.mat33d),
                    mu: wp.float64, lam: wp.float64, kf: wp.float64,
                    energies: wp.array(dtype=wp.float64), stresses: wp.array(dtype=wp.mat33d)):
    i = wp.tid()
    F = Fs[i]
    U, sigma, V = wp.svd3(F)
    logs = wp.vec3d(wp.log(sigma[0]),wp.log(sigma[1]),wp.log(sigma[2]))
    tr = logs[0]+logs[1]+logs[2]
    principal = wp.vec3d()
    for d in range(3):
        principal[d] = (wp.float64(2)*mu*logs[d]+lam*tr)/sigma[d]
    P = U @ wp.diag(principal) @ wp.transpose(V)
    FA = F @ As[i]
    strain = wp.trace(wp.transpose(FA) @ F)-wp.float64(1)
    energies[i] = mu*wp.dot(logs,logs)+wp.float64(.5)*lam*tr*tr+wp.float64(.5)*kf*strain*strain
    stresses[i] = P+wp.float64(2)*kf*strain*FA


class DeviceMaterial:
    def __init__(self,A,params,device='cuda:0'):
        self.device = device
        self.params = params
        self.A = wp.array(np.ascontiguousarray(A),dtype=wp.mat33d,device=device)
        self.F = wp.empty(len(A),dtype=wp.mat33d,device=device)
        self.psi = wp.empty(len(A),dtype=wp.float64,device=device)
        self.P = wp.empty(len(A),dtype=wp.mat33d,device=device)

    def __call__(self,F):
        if not np.isfinite(F).all() or np.any(np.linalg.det(F)<=0):
            raise ValueError('non-positive or non-finite material Jacobian')
        self.F.assign(np.ascontiguousarray(F))
        p = self.params
        wp.launch(response_kernel,dim=len(F),inputs=[self.F,self.A,p.mu,p.lam,p.k_f,self.psi,self.P],device=self.device)
        return self.psi.numpy(),self.P.numpy()
