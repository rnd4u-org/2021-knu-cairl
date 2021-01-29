import numpy as np

def addRow(arr, i):
    arr = np.vstack([arr, [arr[i]]])
    return arr