import numpy as np

def euclidean_distance(x:np.ndarray, y:np.ndarray)->np.float:
    """This is exactly what it says it is"""
    return np.linalg.norm(np.subtract(x,y))
