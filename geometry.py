"""Synthetic curves and area-uniform surfaces used in the final study."""

import numpy as np


def curve(name, t):
    if name == "circle":
        x = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
        a = np.pi / 6
        rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                        [0, np.sin(a), np.cos(a)]])
        return x @ rot.T
    if name == "tennis":
        return np.column_stack((np.cos(t)**3, np.sin(t)**3,
                                np.sqrt(3)/2 * np.sin(2*t)))
    if name == "trefoil":
        return np.column_stack((np.sin(t)+2*np.sin(2*t),
                                np.cos(t)-2*np.cos(2*t), -np.sin(3*t)))
    if name == "sixfolds":
        r = 1 + .3*np.cos(6*t)
        return np.column_stack((r*np.cos(t), r*np.sin(t), .3*np.sin(6*t)))
    raise ValueError(name)


def _curve_grid(name):
    ends = (-np.pi, np.pi) if name == "tennis" else (
        (0, 12*np.pi) if name == "sixfolds" else (0, 2*np.pi))
    t = np.linspace(*ends, 100001)
    x = curve(name, t)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))]
    return t, arc / arc[-1]


def _sphere(n, rng):
    z = 2*rng.random(n) - 1
    theta = 2*np.pi*rng.random(n)
    r = np.sqrt(np.maximum(0, 1-z*z))
    return np.column_stack((r*np.cos(theta), r*np.sin(theta), z))


def _torus(n, rng):
    theta = 2*np.pi*rng.random(n)
    target = 2*np.pi*rng.random(n)
    phi = target.copy()
    for _ in range(8):
        phi -= (phi + .3*np.sin(phi) - target) / (1 + .3*np.cos(phi))
    if np.max(np.abs(phi + .3*np.sin(phi) - target)) >= 1e-12:
        raise ArithmeticError("Torus area-CDF inversion did not converge")
    r = 1 + .3*np.cos(phi)
    return np.column_stack((r*np.cos(theta), r*np.sin(theta), .3*np.sin(phi)))


def sample(name, n, seed, sorted_u=False):
    rng = np.random.default_rng(seed)
    if name in ("circle", "tennis", "trefoil", "sixfolds"):
        t, arc = _curve_grid(name)
        u = rng.random(n)
        return curve(name, np.interp(np.sort(u) if sorted_u else u, arc, t))
    if name == "sphere":
        return _sphere(n, rng)
    if name == "torus":
        return _torus(n, rng)
    raise ValueError(name)


def evaluation_cloud(name, n, seed):
    if name in ("circle", "tennis", "trefoil", "sixfolds"):
        t, arc = _curve_grid(name)
        return curve(name, np.interp(np.linspace(0, 1, n, endpoint=False), arc, t))
    return sample(name, n, seed)
