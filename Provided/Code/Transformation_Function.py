import numpy as np

def transform_point_cloud(xyz,transformation_matrix):
    # xyz = np.reshape(xyz,(-1,3))
    if len(xyz.shape)>1:
        new_xyz = np.ones((xyz.shape[0],xyz.shape[1]+1))
        new_xyz[:,:-1] = xyz
        return np.matmul(new_xyz,np.transpose(transformation_matrix))[:,:-1]
    else:
        new_xyz = np.ones(xyz.shape[0]+1)
        new_xyz[:-1] = xyz
        return np.matmul(new_xyz,np.transpose(transformation_matrix))[:-1]

def transform_pipe(lines,joints,transformation_matrix_list):
    '''transformation_matrix_list: list of 4x4 np.array'''
    for transformation_matrix in transformation_matrix_list:
        for l in lines:
            l.end_points[0] = transform_point_cloud(l.end_points[0],transformation_matrix)
            l.end_points[1] = transform_point_cloud(l.end_points[1],transformation_matrix)
            l.direction = transform_point_cloud(l.direction,transformation_matrix)
        for j in joints:
            if j is not None:
                j.position = transform_point_cloud(j.position,transformation_matrix)
    return lines,joints