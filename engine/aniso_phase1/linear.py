"""Curvature-checked PCG and legacy nonsymmetric Krylov solvers."""
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, bicgstab, gmres
from engine.types import real, vec3
from engine.math.conjugate_gradient import _as_scalar_array
from engine.math.conjugate_gradient import dot_array


def guarded_pcg(A_matvec, b, x, M_inv_matvec, tol, atol, maxiter):
    """Stop before dividing by nonpositive curvature; verify the true residual.

    A successful run certifies its explored directions, not the full spectrum.
    The caller may restart using a positive-definite modified Newton tangent.
    Arrays are confined to free directions by the operator/preconditioner.
    """
    with wp.ScopedDevice(b.device):
        x.zero_()
        r, z, p, Ap = (wp.zeros_like(b) for _ in range(4))
        wp.copy(r, b)
        target = max(atol, tol*np.sqrt(dot_array(b, b)))
        error = np.sqrt(dot_array(r, r))
        if error <= target:
            return 0, error, target, 'converged'
        M_inv_matvec(r, z)
        rz = dot_array(r, z)
        wp.copy(p, z)
        status = 'iteration_limit'
        iterations = 0
        def axpby(a, y, out, alpha, beta):
            wp.launch(combine, dim=3*len(b), inputs=[_as_scalar_array(a), _as_scalar_array(y),
                      _as_scalar_array(out), alpha, beta], device=b.device)
        for iterations in range(1, maxiter+1):
            A_matvec(p, Ap)
            curvature = dot_array(p, Ap)
            if not np.isfinite(curvature) or curvature <= 0:
                status = 'nonpositive_curvature'
                break
            if not np.isfinite(rz) or rz <= 0:
                status = 'invalid_preconditioner'
                break
            alpha = rz/curvature
            axpby(p, x, x, alpha, 1.)
            axpby(Ap, r, r, -alpha, 1.)
            error = np.sqrt(dot_array(r, r))
            if not np.isfinite(error):
                status = 'nonfinite_residual'
                break
            if error <= target:
                status = 'converged'
                break
            M_inv_matvec(r, z)
            rz_new = dot_array(r, z)
            axpby(p, z, p, rz_new/rz, 1.)
            rz = rz_new
        A_matvec(x, Ap)
        axpby(Ap, b, r, -1., 1.)
        error = np.sqrt(dot_array(r, r))
        if not np.isfinite(error):
            status = 'nonfinite_residual'
        if status == 'converged' and error > target*1.01:
            status = 'true_residual_failure'
        return iterations, error, target, status


@wp.kernel
def combine(a: wp.array(dtype=real), y: wp.array(dtype=real), z: wp.array(dtype=real), alpha: real, beta: real):
    i = wp.tid()
    if beta == real(0):
        z[i] = alpha*a[i]
    else:
        z[i] = alpha*a[i]+beta*y[i]


def _solve(A_matvec, b, x, M_inv_matvec, tol, atol, maxiter, method='bicgstab'):
    n = len(b)
    tmp = wp.zeros_like(b)
    def wrap(apply):
        def mv(a, y, z, alpha, beta):
            av = wp.array(ptr=a.ptr, shape=n, dtype=vec3, device=b.device)
            apply(av, tmp)
            wp.launch(combine, dim=3*n, inputs=[_as_scalar_array(tmp), y, z, alpha, beta], device=b.device)
        return LinearOperator((3*n, 3*n), real, b.device, mv)
    A, M = wrap(A_matvec), wrap(M_inv_matvec)
    sb, sx = _as_scalar_array(b), _as_scalar_array(x)
    target = max(atol, tol*float(np.linalg.norm(b.numpy())))
    if method == 'gmres':
        result = gmres(A, sb, sx, tol=tol, atol=atol, maxiter=maxiter, M=M, restart=63, check_every=63, use_cuda_graph=False)
        A_matvec(x, tmp)
        return result[0], float(np.linalg.norm(b.numpy()-tmp.numpy())), target
    result = bicgstab(A, sb, sx, tol=tol, atol=atol, maxiter=maxiter, M=M, check_every=5, use_cuda_graph=False)
    # Verify the true residual: breakdown can be reported as a false convergence.
    A_matvec(x, tmp)
    error = float(np.linalg.norm(b.numpy()-tmp.numpy()))
    target = max(atol, tol*float(np.linalg.norm(b.numpy())))
    if not np.isfinite(error) or error > target:
        x.zero_()
        fallback = gmres(A, sb, sx, tol=tol, atol=atol, maxiter=maxiter, M=M, restart=63, check_every=63, use_cuda_graph=False)
        A_matvec(x, tmp)
        return result[0]+fallback[0], float(np.linalg.norm(b.numpy()-tmp.numpy())), target
    return result[0], error, target


def nonsymmetric_solve(A_matvec, b, x, M_inv_matvec, tol, atol, maxiter, method="bicgstab"):
    # Warp 1.10 TiledDot records some launches on the current device.
    with wp.ScopedDevice(b.device):
        return _solve(A_matvec, b, x, M_inv_matvec, tol, atol, maxiter, method)
