import numpy as np

def isin(ar, ar_collec):
    return (ar == ar_collec).all(axis=1).any()
