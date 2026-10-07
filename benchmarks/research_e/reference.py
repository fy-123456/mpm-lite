"""Analytic references independent of the E matrix assembly."""
import numpy as np


def consolidation(x, time, modulus=40., storage=.001, alpha=1., conductivity=.01, load=.1, length=1., terms=400):
    x = np.asarray(x)
    j = np.arange(terms); k = (j+.5)*np.pi/length
    p0 = alpha*load/(storage*modulus+alpha**2)
    coeff = 4*(-1.)**j/(np.pi*(2*j+1))*np.exp(-conductivity/(storage+alpha**2/modulus)*k*k*time)
    p = p0*np.sum(coeff*np.cos(x[..., None]*k), axis=-1)
    integral = p0*np.sum(coeff*np.sin(x[..., None]*k)/k, axis=-1)
    u = (alpha*integral-load*x)/modulus
    return p, u


def consolidation_average(n, time, **kwargs):
    # Cell integral via displacement primitive; exact Fourier antiderivative.
    length = kwargs.get('length', 1.)
    modulus = kwargs.get('modulus', 40.); alpha = kwargs.get('alpha', 1.); load = kwargs.get('load', .1)
    edges = np.linspace(0, length, n+1)
    _, u = consolidation(edges, time, **kwargs)
    primitive = (modulus*u+load*edges)/alpha
    return np.diff(primitive)/(length/n)


def sine_fields(x):
    x, y = x; pi = np.pi
    f = np.sin(pi*x)*np.sin(pi*y)
    grad = pi*np.array([np.cos(pi*x)*np.sin(pi*y), np.sin(pi*x)*np.cos(pi*y)])
    H = pi*pi*np.array([[-f, np.cos(pi*x)*np.cos(pi*y)], [np.cos(pi*x)*np.cos(pi*y), -f]])
    return f, grad, H


def biot_manufactured(material, K, storage, alpha=1., amplitude=.0001, pressure=.1):
    c = amplitude*np.array([1., .5]); a = material.fiber_direction[:2]; A = np.outer(a,a)
    def displacement(x): return c*sine_fields(x)[0]
    def pressure_field(x): return pressure*sine_fields(x)[0]
    def force(x):
        _, grad, H = sine_fields(x)
        divstress = material.mu*c*np.trace(H)+(material.lam+material.mu)*H @ c+4*material.k_f*(a @ c)*A @ H @ a
        return -divstress+alpha*pressure*grad
    def source(x, time):
        f, grad, H = sine_fields(x)
        return alpha*(c @ grad)+storage*pressure*f-time*pressure*np.sum(K*H)
    return displacement, pressure_field, force, source
