import numpy as np
from numpy.polynomial import polynomial as poly
from lwe_constants import N, Q, K_ERROR

class PolyOps:
    """
    A class to handle polynomial arithmetic in the ring R_q = Z_q[x] / (x^N + 1).
    Performs reduction modulo Q and modulo (x^N + 1).
    """
    def __init__(self, N, Q):
        self.N = N
        self.Q = Q

    def reduce_mod_q(self, p):
        """Reduces all coefficients of a polynomial modulo Q."""
        return p % self.Q

    def poly_mul(self, p1, p2):
        """
        Multiplies two polynomials p1 and p2 and reduces the result
        modulo (x^N + 1) and modulo Q.
        """
        # 1. Standard polynomial multiplication (convolution)
        p_long = poly.polymul(p1, p2)
        
        # 2. Reduction modulo (x^N + 1)
        p_reduced = np.zeros(self.N, dtype=int)
        p_reduced[:self.N] = p_long[:self.N]
        
        for i in range(self.N, len(p_long)):
            idx = i - self.N
            p_reduced[idx] = (p_reduced[idx] - p_long[i])
            
        # 3. Reduction modulo Q
        return self.reduce_mod_q(p_reduced).astype(int)

    def sample_A(self):
        """Generates a public polynomial A with uniform random coefficients in Z_Q."""
        return np.random.randint(0, self.Q, size=self.N, dtype=int)

    def sample_small(self, k=K_ERROR):
        """
        Samples a 'small' polynomial (secret key s or error e) from a centered 
        binomial distribution approximation (CBD).
        """
        p = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            b1 = np.random.randint(0, 2, size=k)
            b2 = np.random.randint(0, 2, size=k)
            p[i] = np.sum(b1) - np.sum(b2)
        return p

# Instantiate the PolyOps helper for use in r-lwe.py
PO = PolyOps(N, Q)