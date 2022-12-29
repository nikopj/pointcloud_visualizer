import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform

def euler2Quat(rot):
    r = Rotation.from_euler('xyz',rot,degrees=True)
    q = Rotation.as_quat(r)
    return q

def quat2Euler(q):
    r = Rotation.from_quat(q)
    ang = Rotation.as_euler(r,'xyz',degrees='true')
    return ang

def transRot(V,rot,t,asform="quaternion"):
    # translates and rotates a given point cloud by specified parameters
    # V is a point cloud, 
    # rot is a rotation, which can be given in 3 forms determined by asform
    #   1) a quaternion (versor) rotation,
    #   2) a rotation matrix
    #   3) a set of Euler angles
    # t is a 3D translation
    
    # first map all rotations to matrices using scipy's nice little rotation module!
    if asform == "quaternion":
        # rot is a quaternion
        r = np.asarray(Rotation.as_matrix(Rotation.from_quat(rot)))
    elif asform == "matrix":
        # rot is a rotation matrix
        r = rot
    elif asform == "angles":
        # rot is 3 angles -- x, y, z
        r = np.asarray(Rotation.as_matrix(Rotation.from_euler('xyz',rot,degrees=True)))
    
    N = V.shape[0]
    # it is just a matrix multiply and addition
    V = np.tile(t,(N,1))+np.transpose((r@np.transpose(V)))
    
    return V 

def pcclip(pc, radius, ord=np.inf, normalize=False):
    # pc: pointcloud size N x num_features. 
    # considers first 3 features xyz coords
    norms = np.linalg.norm(pc[:,:3], ord=ord, axis=1)
    maxnorm = np.max(norms) if normalize else 1.0
    inds =  (norms / maxnorm) <= radius
    return pc[inds, :], inds
