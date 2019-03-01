import numpy as np

def percentage_confusion_matrix(confMat):
    return np.around((confMat / np.sum(confMat,axis=1)[:,None])*100,2)