# Importing useful libraries
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

# Import classes from Clustering.py and Extracting_Data.py
from Scripts.Clustering import Clustering_Vectors
from Scripts.Extracting_Data import Extraction

# Define a class for functions that are required for 
# plotting the required data
class Surface_Plotting():

    # Function for returning x, y, and z as lists
    # Points that constitute a surface with the 
    # provided direction vectors 
    # (x,y,z) = (x0,y0,z0) + t(mx,my,mz)
    # dx - Step for changing the points (t)
    # dir_vec - Direction vector slopes (mx,my,mz)
    def make_surface(self, dx, dir_vec):
        
        # List of points
        xs = []
        ys = []
        zs = []
        
        for i in range(len(dx)):

            # (x,y,z) = (x0,y0,z0) + t(mx,my,mz)
            temp_x = x_min + dx[i]*dir_vec[0]
            temp_y = y_min + dx[i]*dir_vec[1]
            temp_z = z_min + dx[i]*dir_vec[2]
        
            xs.append(temp_x)
            ys.append(temp_y)
            zs.append(temp_z)

        return xs, ys, zs


filename = "../Data/pipelist_updated_centerlines_original.txt"

clustering = Clustering_Vectors()
extracting = Extraction()
plotting = Surface_Plotting()

line_list = extracting.extracting_line_points(filename)

line_data = []
line_data.append(line_list[1:])

# Starting and ending points of centerlines
x0, y0, z0 = extracting.individual_components(5, 6, 7, line_data)
x1, y1, z1 = extracting.individual_components(8, 9, 10, line_data)

# Minimum and maximum values from the set of 
# starting and ending points
x_min, x_max = extracting.min_max(x0, x1)
y_min, y_max = extracting.min_max(y0, y1)
z_min, z_max = extracting.min_max(z0, z1)

length_data = len(x0)

# Starting points list (x,y,z)
p0 = []
for i in range(length_data):
    p0.append((x0[i], y0[i], z0[i]))

# Ending points list (x,y,z)
p1 = []
for i in range(length_data):
    p1.append((x1[i], y1[i], z1[i]))

# Vectors of each centerline - starting point 
# to ending point
Vectors = np.zeros((length_data,3))

# Normalized vectors
M = np.zeros((length_data,3))

for i in range(length_data):

    # Length
    l = math.sqrt((p0[i][0]-p1[i][0])**2 + (p0[i][1]-p1[i][1])**2 + (p0[i][2]-p1[i][2])**2)
    
    M[i,0] = (p1[i][0]-p0[i][0])/l
    M[i,1] = (p1[i][1]-p0[i][1])/l
    M[i,2] = (p1[i][2]-p0[i][2])/l
    
    Vectors[i, :] = M[i, :] * l
    
#centerlines.hierarchical_clustering(M)

# K-means clustering
centers_kmeans = clustering.Kmeans_clustering(M)

# List of all points
x = x0.copy()
x.extend(x1)
y = y0.copy()
y.extend(y1)
z = z0.copy()
z.extend(z1)


fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.scatter3D(x, y, z, cmap='Greens')

dx = np.linspace(0, 10, 1000)

# Surface points according to K means clustered 
# direction vectors
xs_1, ys_1, zs_1 = plotting.make_surface(dx, centers_kmeans[0])
xs_2, ys_2, zs_2 = plotting.make_surface(dx, centers_kmeans[1])
xs_3, ys_3, zs_3 = plotting.make_surface(dx, centers_kmeans[2])

# Plotting the surface points
ax.scatter3D(xs_1, ys_1, zs_1, cmap='Greens')
ax.scatter3D(xs_2, ys_2, zs_2, cmap='Greens')
ax.scatter3D(xs_3, ys_3, zs_3, cmap='Greens')

filename_allPoints = "../Data/lines_original.txt"
lines = extracting.extracting_line_points(filename_allPoints)

allData = []
allData.append(lines[0:])

x_data = extracting.extracting_components(allData, 0)
y_data = extracting.extracting_components(allData, 1)
z_data = extracting.extracting_components(allData, 2)

ax.scatter3D(x_data, y_data, z_data, cmap='Greens')
plt.show()