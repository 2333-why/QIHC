"""Deterministic representation normalization."""
def dense(value, n, path):
    if not isinstance(value, dict):
        return value
    out = [0]*n
    for key, coefficient in value.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key or not 0 <= int(key) < n:
            raise ValueError(path+': sparse index must be a canonical string integer in 0..n-1')
        if type(coefficient) is not int:
            raise ValueError(path+': coefficient must be integer, not float/bool')
        out[int(key)] = coefficient
    return out
