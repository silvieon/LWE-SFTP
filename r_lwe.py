import numpy as np
from poly_ops import PO # Get the pre-instantiated PolyOps object
from lwe_constants import N, Q, K_ERROR, a

def KeyGen(A):

    # Alice or Bob's Key Generation.
    # A is the public polynomial (shared).
    # Returns (s, P), where s is the secret key and P is the public key (b = a*s + e).

    s = PO.sample_small(k=K_ERROR)
    e = PO.sample_small(k=K_ERROR)
    a_times_s = PO.poly_mul(A, s)
    b = PO.reduce_mod_q(a_times_s + 2 * e)
    return s, b # P is b

def HelpRec(v):

    # Alice's Reconciliation Hint Generator (HelpRec).
    # Input: v = raw shared secret (v = B_B * s_A + e_A_prime).
    # Output: h (The hint/correction factor, polynomial of 0s and 1s).

    h = np.zeros(N, dtype=int)
    for i in range(N):
        v_i = v[i] % Q
        
        # ZONE 2: Close to Q/2 -> [Q/4, 3*Q/4)
        if v_i >= a and v_i < (3 * a):
            h[i] = 1 
        else:
            h[i] = 0 # ZONE 1: Close to 0
            
    return h

def Rec(w, h):

    # Bob's Shared Secret Recovery (Rec).
    # Input: w = raw shared secret K_A, K_B.
    #        h = Alice's reconciliation hint.
    # Output: K (The final shared key, a polynomial of 0s and 1s).

    K = np.zeros(N, dtype=int)
    
    for i in range(N):
        w_i = w[i] % Q
        h_i = h[i]
        
        # 1. Apply hint: shift w_i by Q/2 if h_i is 1
        if h_i == 1:
            w_i = (w_i + (Q // 2)) % Q
            
        # 2. Key recovery: Check if the result w_i is closer to 0 or Q/2 (mod Q)
        # Thresholds are ALPHA (Q/4) and Q - ALPHA (3*Q/4)
        if w_i < a or w_i >= (Q - a):
            K[i] = 0 # Closer to 0 mod Q
        else:
            K[i] = 1 # Closer to Q/2 mod Q (Indicates a failure or the intended 1-bit output)

    return K