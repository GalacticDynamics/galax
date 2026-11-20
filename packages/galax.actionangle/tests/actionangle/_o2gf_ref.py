"""NumPy reference for the O2GF linear systems.

The three functions below are adapted from gala
(gala/dynamics/actionangle/actionangle_o2gf.py, MIT license,
Copyright (c) the gala developers): `check_angle_sampling`, `_action_prepare`
and `_angle_prepare`. Warnings are removed, the n-vectors are passed in, and
`aa` is the gala-shaped (6, T) array of toy actions then toy angles.
"""

import numpy as np


def check_angle_sampling(nvecs, angles):
    failed_nvecs = []
    failures = []
    for vec in nvecs:
        X = (angles * vec[:, None]).sum(axis=0)
        diff = float(np.abs(X.max() - X.min()))
        if diff < (2.0 * np.pi):
            failed_nvecs.append(vec.tolist())
            failures.append(0)
        elif (diff / len(X)) > np.pi:
            failed_nvecs.append(vec.tolist())
            failures.append(1)
    return np.array(failed_nvecs), np.array(failures)


def action_prepare(aa, nvecs):
    n = len(nvecs) + 3
    b = np.zeros(shape=(n,))
    A = np.zeros(shape=(n, n))
    A[:3, :3] = aa.shape[1] * np.identity(3)
    actions = aa[:3]
    angles = aa[3:]
    C_T = 2.0 * nvecs.T * np.sum(np.cos(np.dot(nvecs, angles)), axis=-1)
    A[:3, 3:] = C_T
    A[3:, :3] = C_T.T
    cosv = np.cos(np.dot(nvecs, angles))
    A[3:, 3:] = 4.0 * np.dot(nvecs, nvecs.T) * np.einsum("it, jt->ij", cosv, cosv)
    b[:3] = np.sum(actions, axis=1)
    b[3:] = 2 * np.sum(np.dot(nvecs, actions) * np.cos(np.dot(nvecs, angles)), axis=1)
    return A, b


def angle_prepare(aa, t, nvecs):
    """`aa[3:]` must already be unwrapped, as gala does with `np.unwrap`."""
    angles = aa[3:]
    nv = len(nvecs)
    n = 3 + 3 + 3 * nv
    b = np.zeros(shape=(n,))
    A = np.zeros(shape=(n, n))
    A[:3, :3] = aa.shape[1] * np.identity(3)
    A[:3, 3:6] = A[3:6, :3] = np.sum(t) * np.identity(3)
    A[3:6, 3:6] = np.sum(t * t) * np.identity(3)
    A[6 : 6 + nv, 0] = -2.0 * np.sum(np.sin(np.dot(nvecs, angles)), axis=1)
    A[6 + nv : 6 + 2 * nv, 1] = A[6 : 6 + nv, 0]
    A[6 + 2 * nv : 6 + 3 * nv, 2] = A[6 : 6 + nv, 0]
    A[6 : 6 + nv, 3] = -2.0 * np.sum(t[None, :] * np.sin(np.dot(nvecs, angles)), axis=1)
    A[6 + nv : 6 + 2 * nv, 4] = A[6 : 6 + nv, 3]
    A[6 + 2 * nv : 6 + 3 * nv, 5] = A[6 : 6 + nv, 3]
    sinv = np.sin(np.dot(nvecs, angles))
    SdotST = np.einsum("it, jt->ij", sinv, sinv)
    A[6 : 6 + nv, 6 : 6 + nv] = A[6 + nv : 6 + 2 * nv, 6 + nv : 6 + 2 * nv] = A[
        6 + 2 * nv : 6 + 3 * nv, 6 + 2 * nv : 6 + 3 * nv
    ] = 4 * SdotST
    A[:6, :] = A[:, :6].T
    b[:3] = np.sum(angles.T, axis=0)
    b[3:6] = np.sum(t[:, None] * angles.T, axis=0)
    b[6 : 6 + nv] = -2.0 * np.sum(angles[0] * np.sin(np.dot(nvecs, angles)), axis=1)
    b[6 + nv : 6 + 2 * nv] = -2.0 * np.sum(
        angles[1] * np.sin(np.dot(nvecs, angles)), axis=1
    )
    b[6 + 2 * nv : 6 + 3 * nv] = -2.0 * np.sum(
        angles[2] * np.sin(np.dot(nvecs, angles)), axis=1
    )
    return A, b
