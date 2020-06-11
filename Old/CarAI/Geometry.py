import numpy as np


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def find_intersection(a, b, c, d):
    xdiff = (a[0] - b[0], c[0] - d[0])
    ydiff = (a[1] - b[1], c[1] - d[1])
    div = det(xdiff, ydiff)

    ## If this determinant is zero then the lines do not intersect
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



def check_orientation(a, b, c):
    """ This returns the orientation of a triplet of 2D points
        0: colinear
        1: clockwise
        2: counter-clockwise
    """
    val = (b[1]-a[1])*(c[0]-b[0])-(b[0]-a[0])*(c[1]-b[1])
    if   val == 0: return 0
    elif val >  0: return 1
    else:          return 2

def on_segment(a, b, c):
    """ Given three collinear points this funciton
        checks that point c lies on ab by seeing if they are within the ranges
    """
    if c[0]<=max(a[0],b[0]) and c[0]>=min(a[0],b[0]) and \
       c[1]<=max(a[1],b[1]) and c[0]>=min(a[1],b[1]):
       return True
    return False

def check_segment_intersect(a,b,c,d):
    """ The function that check if the line segment joining ab
        intersects with the line segment bc
    """

    # Find the 4 orientations required for
    # the general and special cases
    o1 = check_orientation(a, b, c)
    o2 = check_orientation(a, b, d)
    o3 = check_orientation(c, d, a)
    o4 = check_orientation(c, d, b)

    # General case, if the orientations change, then they intersect
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases of colinearity
    if ((o1 == 0) and on_segment(a, b, c)): return True
    if ((o2 == 0) and on_segment(a, b, d)): return True
    if ((o3 == 0) and on_segment(c, d, a)): return True
    if ((o4 == 0) and on_segment(c, d, b)): return True

    # If none of the cases
    return False

def rotate_2d_vec( vector, angle):
    """ Use numpy to create a rotation matrix then we take the dot product
    """
    c, s    = np.cos(angle), np.sin(angle)
    rot_mat = np.matrix([[c,-s],[s,c]])
    return np.asarray(np.dot( rot_mat, vector )).squeeze()
