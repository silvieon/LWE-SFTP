import numpy as np

# --- R-LWE CONFIGURATION ---
N = 2048     # Degree of the polynomial ring (x^n + 1). Must be a power of 2.
Q = 8380417    # Modulus (prime number).
K_ERROR = 2 # Security parameter for error sampling (CBD approximation).

# Helper constants for the key reconciliation step
a = Q / 4