import numpy as np


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def find_intersection(a, b, c, d):

    xdiff = (a[0] - b[0], c[0] - d[0])
    ydiff = (a[1] - b[1], c[1] - d[1])
    div = det(xdiff, ydiff)

    ## If this determinant is zero then the lines are parallel
    if div == 0:
        return False, 0

    ## We can then calculate the x value of the line
    delta = (det(a, b), det(c, d))
    x = det(delta, xdiff) / div

    ## We check if this x value lies inside the line segments
    if (min(a[0],b[0])-1e-8<=x<=max(a[0],b[0])+1e-8) and (min(c[0],d[0])-1e-8<=x<=max(c[0],d[0])+1e-8):
        y = det(delta, ydiff) / div

        if (min(a[1],b[1])-1e-8<=y<=max(a[1],b[1])+1e-8) and (min(c[1],d[1])-1e-8<=y<=max(c[1],d[1])+1e-8):
            return True, np.array([x, y])

    return False, 0


def rotate_2d_vec( vector, angle):
    """ Use numpy to create a rotation matrix then we take the dot product
    """
    c, s    = np.cos(angle), np.sin(angle)
    rot_mat = np.matrix([[c,-s],[s,c]])
    return np.asarray(np.dot( rot_mat, vector )).squeeze()
