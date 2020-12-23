import pandas as pd
import numpy as np

import matplotlib

from math import log2, log

def kl_divergence(p, q, base="log2"):
    """
    Compute Kullback Leibler Divergence for the discrete distributions p and q.
    If base=None: log base-2 to ensure the result has units in bits.
    If base=log: result has units in nats (equivalent to Scipy Relative Entropy)
    """
    if base == "log":
        return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)))
    elif base == "log2":
        return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


def js_divergence(p, q):
    """
    Compute Jensen-Shannon Divergence fot the discrete distributions q and q
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

