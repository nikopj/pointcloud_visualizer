import numpy as np

def pcclip(pc, radius, ord=np.inf, normalize=False):
    # pc: pointcloud size N x num_features. 
    # considers first 3 features xyz coords
    norms = np.linalg.norm(pc[:,:3], ord=ord, axis=1)
    maxnorm = np.max(norms) if normalize else 1.0
    inds =  (norms / maxnorm) <= radius
    return pc[inds, :], inds


