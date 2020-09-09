import numpy as np
import math
import statistics


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normalized_pts = []
    for pt in pts:
        normalized_pts.append([(pt[0] - pp[0])/focal, (pt[1] - pp[1])/focal])

    return np.array(normalized_pts)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnormalized_pts = []
    for pt in pts:
        unnormalized_pts.append([(pt[0] * focal) + pp[0], (pt[1] * focal) + pp[1]])

    return np.array(unnormalized_pts)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:-1, :-1]
    T = EM[:-1, -1]
    foe = [T[0]/T[2], T[1]/T[2]]
    tZ = T[2]

    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    rotated_pts = []
    for i, pt in enumerate(pts):
        ray = np.array([pt[0], pt[1], 1])
        rotated = R.dot(ray)
        rotated_pts.append([rotated[0]/rotated[2], rotated[1]/rotated[2]])

    return np.array(rotated_pts)


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = ((foe[1]-p[1])/(foe[0]-p[0]))
    n = ((foe[0]*p[1])-(foe[1]*p[0]))/(foe[0] - p[0])

    def distance_from_line(x, y):
        return abs((m*x + n - y)/math.sqrt(m**2 + 1))

    corresponding_p_ind = -1
    corresponding_p_rot = []
    min_dist = None

    for i, pt in enumerate(norm_pts_rot):
        dist = distance_from_line(pt[0], pt[1])
        if not min_dist or dist < min_dist:
            corresponding_p_ind = i
            corresponding_p_rot = pt
            min_dist = dist

    return corresponding_p_ind, corresponding_p_rot


def calc_dist(p_curr, p_rot, foe, tZ):
    Zx = tZ*((foe[0] - p_rot[0])/(p_curr[0] - p_rot[0]))
    Zy = tZ*((foe[1] - p_rot[1])/(p_curr[1] - p_rot[1]))

    x_diff = abs(p_rot[0] - p_curr[0])
    y_diff = abs(p_rot[1] - p_curr[1])

    return (x_diff/(x_diff+y_diff))*Zx + (y_diff/(x_diff+y_diff))*Zy

    # diff = abs(foe[0] - p_curr[0]/(foe[1] - p_curr[1]))
    # return (abs(math.sin(diff)/diff))*Zx + (1 - abs(math.sin(diff)/diff))*Zy