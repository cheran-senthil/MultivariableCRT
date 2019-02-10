"""Multivariable Chinese Remainder Theorem"""

from math import gcd

import numpy as np


def egcd(a, m):
    """Extended GCD"""
    if a == 0:
        return (m, 0, 1)
    g, y, x = egcd(m % a, a)
    return (g, x - (m // a) * y, y)


def modinv(a, m):
    """Find Modular Inverse"""
    amodm = a % m
    g, x, _ = egcd(amodm, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def pivot(A, m):
    """Finds the pivot of A and m"""
    length = len(A)
    result = [0] * length
    for i in range(length):
        for j in range(length):
            if gcd(A[i][j], m[i]) == 1:
                result[i] = j
    return result


def is_sol(A, x, b, m):
    """Checks if Ax = b mod m"""
    ax_b = np.matmul(np.array(A), np.array(x)) - np.array(b)
    for i, mod in enumerate(m):
        if ax_b[i] % mod != 0:
            return False
    return True


def mcrt(A, b, m):
    """Returns for x in Ax = b mod m"""
    eqn_cnt = len(A)
    piv = pivot(A, m)
    x = [0] * eqn_cnt
    m_prod = 1

    for i in range(eqn_cnt):
        tot = sum(A[i][k] * x[k] for k in range(eqn_cnt))
        tmp = (modinv(m_prod * A[i][piv[i]], m[i]) * (b[i] - tot)) % m[i]
        x[piv[i]] = x[piv[i]] + tmp * m_prod
        m_prod *= m[i]

    return x
