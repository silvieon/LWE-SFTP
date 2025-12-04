import numpy as np
from poly_ops import PO # Get the pre-instantiated PolyOps object
from lwe_constants import Q, K_ERROR

# Helper constants for the key reconciliation step
a = np.floor(Q / 4).astype(int) 

Q_HALF = (Q - 1) // 2

def KeyGen(A):
    """
    Alice or Bob's Key Generation.
    A is the public polynomial (shared).
    Returns (s, P), where s is the secret key and P is the public key (b = a*s + e).
    """
    s = PO.sample_small(k=K_ERROR)
    e = PO.sample_small(k=K_ERROR)
    a_times_s = PO.poly_mul(A, s)
    b = PO.reduce_mod_q(a_times_s + 2 * e)
    return s, b

# ==============================================================================
# 1. SIGNAL FUNCTION (Sig) - Implemented in HelpRec
# Sig(v) = 0 if v is in the central region E = [-T, T], 1 otherwise.
# ==============================================================================
def HelpRec(v):
    """
    Alice's Reconciliation Hint Generator (HelpRec), now implemented as Sig(v).
    Input: v = raw shared secret (v = B_B * s_A + 2*e_A_prime).
    Output: hint/correction factor 'w', polynomial of 0s and 1s.
    """
    # 1. Center the coefficients of v into the range [-(q-1)/2, (q-1)/2]
    # We use Q_MODULUS here for the centering operation.
    v_centered = PO.reduce_mod_q(v + Q_HALF) - Q_HALF

    # 2. Apply the Sig function threshold: 
    # Sig(v) = 1 if |v_centered| > T (outside of E), 0 otherwise.
    w = np.where(np.abs(v_centered) > a, 1, 0)
    
    # h is the reconciliation signal 'w'
    return w

# ==============================================================================
# 2. MOD 2 RECONCILIATION FUNCTION (Mod2) - Implemented in Rec
# Mod_2(v, w) = (v + w * Q_HALF) mod q mod 2
# ==============================================================================
def Rec(k_raw, w_signal):
    """
    Bob's Shared Secret Recovery (Rec), now implemented as Mod2(k_A, w).
    Input: k = raw shared secret (k_A or k_B)
           w = Alice's reconciliation hint (w from Sig function).
    Output: K (The final shared key sk_A or sk_B, a vector of 0s and 1s).
    """

    # 1. Apply the reconciliation formula: (k_raw + w_signal * Q_HALF) mod q
    # Q_HALF is the constant (q-1)/2
    reconciled_intermediate = PO.reduce_mod_q(k_raw + w_signal * Q_HALF)

    # 2. Apply the final mod 2 operation element-wise to get the key stream K
    K = np.mod(reconciled_intermediate, 2)

    return K