import open3d as o3d 
import numpy as np
from scipy.spatial import KDTree

def PCA(data, correlation=False, sort=True):
    
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    
    H = np.dot(normal_data.T, normal_data)
    
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def main():

    filename = "..\Data\lines_original.txt"
    points = np.loadtxt(filename, delimiter=' ')[:, 0:3] # Import TXT data to np.array, here only import 3 columns
    print('total points number is:', points.shape[0])

    #        
    w, v = PCA(points) # PCA method get the corresponding feature value and feature vector
    point_cloud_vector = v[:, 0] #     mainly direction corresponding to the characteristic vector corresponding to the maximum feature value
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # Three feature vectors constitute three coordinate axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))

    #   Calculate the method of each point
    leafsize = 32   # Switch to the minimum number of violent search
    KDTree_radius = 0.1 # Set the neighbor area radius
    tree = KDTree(points, leafsize=leafsize) # Build kdtree
    radius_neighbor_idx = tree.query_ball_point(points, KDTree_radius) # Get the neighboring index of each point
    normals = [] # Define an empty list

    # ------------- Looking for the normal ---------------
    # First, look for points in the neighborhood
    for i in range(len(radius_neighbor_idx)):
        neighbor_idx = radius_neighbor_idx[i] # Get the neighboring point index of the i-th point, neighboring points include yourself
        neighbor_data = points[neighbor_idx] # Get neighbor points, there is no need to normalize when seeking neighboring normal, and it will be normalized in the PCA function.
        eigenvalues, eigenvectors = PCA(neighbor_data) # Do PCA adjacent points to get feature values ​​and feature vectors
        normals.append(eigenvectors[:, 2]) #                  
    # ------------ Facade looks over ---------------
    normals = np.array(normals, dtype=np.float64) # Put the normal line in Normals
    # O3d.geometry.PointCloud, return PointCloud type
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    # Add a normal to the PointCloud object
    pc_view.normals = o3d.utility.Vector3dVector(normals)
    # Visualization
    o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)

if __name__ == '__main__':
    main()