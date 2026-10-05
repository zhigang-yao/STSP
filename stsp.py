"""Shared fixed-reference STSP and MS-STSP algorithms; arrays are N by D."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

VERSION = 'final-radius-1.0'

class Reference:
    def __init__(self, x):
        self.x = np.array(x, dtype=float, copy=True)
        if self.x.ndim != 2 or not len(self.x) or not np.isfinite(self.x).all():
            raise ValueError('Reference must be a finite nonempty N by D array')
        self.x.flags.writeable = False
        self.nn = NearestNeighbors(algorithm='kd_tree').fit(self.x)


class BatchProjector:
    """Batched d=1 radius projection used by the Dentate Gyrus fits."""
    def __init__(self, ref):
        self.ref = ref
        self.x = ref.x
        self.n, self.d = self.x.shape
        self.tri = np.triu_indices(self.d)
        self.products = self.x[:, self.tri[0]] * self.x[:, self.tri[1]]

    def update(self, y, h, block_size=128):
        out = y.copy()
        for start in range(0, len(y), block_size):
            z = y[start:start+block_size]
            ids = self.ref.nn.radius_neighbors(z, radius=h,
                                               return_distance=False)
            counts = np.asarray([len(v) for v in ids])
            good = counts >= 3
            if not good.any():
                continue
            rows = np.repeat(np.arange(len(z)), counts)
            cols = np.concatenate(ids)
            val = np.repeat(1 / np.maximum(counts, 1), counts)
            w = sparse.csr_matrix((val, (rows, cols)),
                                  shape=(len(z), self.n))
            mu = w @ self.x
            second = w @ self.products
            cov = np.empty((len(z), self.d, self.d))
            cov[:, self.tri[0], self.tri[1]] = second
            cov[:, self.tri[1], self.tri[0]] = second
            cov -= mu[:, :, None] * mu[:, None, :]
            _, vec = np.linalg.eigh(cov[good])
            tangent = vec[:, :, -1]
            delta = z[good] - mu[good]
            block = out[start:start+len(z)]
            block[good] = (mu[good] + tangent *
                           np.einsum('ij,ij->i', tangent, delta)[:, None])
        return out

def update(y, ref, h, d, block_size=512):
    """Project onto the local d-dimensional tangent plane inside radius h."""
    y = np.asarray(y, float)
    D = ref.x.shape[1]
    if y.ndim != 2 or y.shape[1] != D or not 1 <= d < D:
        raise ValueError('Require matching ambient dimension and 1 <= d < D')
    if not np.isfinite(h) or h <= 0:
        raise ValueError('Radius h must be finite and positive')
    out = y.copy(); counts = np.zeros(len(y), int)
    for a in range(0, len(y), block_size):
        z = y[a:a+block_size]
        ids = list(ref.nn.radius_neighbors(z, radius=h, return_distance=False))
        ns = np.array([len(v) for v in ids])
        counts[a:a+len(z)] = ns
        good = ns >= d+2
        if D > 3:
            for i in np.flatnonzero(good):
                x = ref.x[ids[i]]; mu = x.mean(axis=0)
                _, _, vt = np.linalg.svd(x-mu, full_matrices=False)
                u = vt[:d].T
                out[a+i] = mu + u @ (u.T @ (z[i]-mu))
            continue
        rows = np.repeat(np.arange(len(z)), ns)
        cols = np.concatenate(ids).astype(int)
        denom = np.maximum(ns, 1)
        mu = np.column_stack([np.bincount(rows, weights=ref.x[cols,j],
                              minlength=len(z))/denom for j in range(D)])
        cov = np.empty((len(z), D, D))
        for j in range(D):
            for ell in range(j,D):
                val = np.bincount(rows, weights=ref.x[cols,j]*ref.x[cols,ell],
                                  minlength=len(z))/denom - mu[:,j]*mu[:,ell]
                cov[:,j,ell] = val; cov[:,ell,j] = val
        _, u = np.linalg.eigh(cov); u = u[:,:,-d:]
        v = mu + np.einsum('nij,nj->ni',u,np.einsum('nij,ni->nj',u,z-mu))
        v[~good] = z[~good]; out[a:a+len(z)] = v
    return out, counts

def weights(hs):
    hs = np.asarray(hs,float)
    if hs.ndim != 1 or len(hs)<2 or np.any(hs<=0) or len(np.unique(hs))<2:
        raise ValueError('At least two distinct positive radii required')
    # A scaled abscissa improves conditioning without changing the intercept.
    A = np.vstack([np.ones(len(hs)), (hs/hs.max())**2])
    return A.T @ np.linalg.solve(A@A.T, np.array([1.,0.]))

@dataclass
class Result:
    y: np.ndarray
    iterations: int
    reason: str
    moves: list
    history: list

def iterate(x, h, d, tmax, s, reference=None,
            tolerance=1e-3, stop=True):
    if not isinstance(tmax, (int, np.integer)) or tmax<1 or not np.isfinite(s) or s<=0:
        raise ValueError('Integer Tmax >= 1 and finite s > 0 required')
    if not np.isfinite(tolerance) or tolerance<=0: raise ValueError('Finite positive tolerance required')
    ref = Reference(x if reference is None else reference)
    y = np.array(x,float,copy=True); moves=[]; history=[]; reason='Tmax'
    for t in range(1,tmax+1):
        yn,_=update(y,ref,h,d)
        moves.append(float(np.linalg.norm(yn-y,axis=1).mean()/s))
        y=yn; history.append(y.copy())
        if stop and moves[-1]<tolerance: reason='threshold'; break
    return Result(y,t,reason,moves,history)

def aggregate(histories, moves, hs, tolerance=1e-3):
    if not np.isfinite(tolerance) or tolerance<=0: raise ValueError('Finite positive tolerance required')
    w=weights(hs); m=np.max(np.asarray(moves),axis=0)
    hits=np.flatnonzero(m<tolerance); t=int(hits[0]+1) if len(hits) else len(m)
    y=np.tensordot(w,np.stack([hist[t-1] for hist in histories]),axes=(0,0))
    return Result(y,t,'threshold' if len(hits) else 'Tmax',m[:t].tolist(),[]),w

def multiscale(x, hs, d, tmax, s, tolerance=1e-3):
    branches=[iterate(x,h,d,tmax,s,tolerance=tolerance,stop=False) for h in hs]
    return aggregate([b.history for b in branches],[b.moves for b in branches],hs,tolerance)

def fit_stsp(x,h,d,tmax,s,tolerance=1e-3):
    return iterate(x,h,d,tmax,s,tolerance=tolerance)

def fit_ms_stsp(x,hs,d,tmax,s,tolerance=1e-3):
    return multiscale(x,hs,d,tmax,s,tolerance=tolerance)
