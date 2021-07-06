import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math

from ExtractingData import extraction

centerlinesFileName = "../Data/centerlines.txt"

extracting = extraction()

line_list = extracting.extractingLinePoints(centerlinesFileName)

line_data = []
line_data.append(line_list[1:])

x0, y0, z0 = extracting.individualComponents(5, 6, 7, line_data)
x1, y1, z1 = extracting.individualComponents(8, 9, 10, line_data)

grp, lineNo, rad = extracting.individualComponents(1, 2, 3, line_data)

length_data = len(x0)

p0 = []
for i in range(length_data):
    p0.append((x0[i], y0[i], z0[i]))

p1 = []
for i in range(length_data):
    p1.append((x1[i], y1[i], z1[i]))

Vectors = np.zeros((length_data,3))
M = np.zeros((length_data,3))


for i in range(length_data):

    l = math.sqrt((p0[i][0]-p1[i][0])**2 + (p0[i][1]-p1[i][1])**2 + (p0[i][2]-p1[i][2])**2)
    
    M[i,0] = (p1[i][0]-p0[i][0])/l
    M[i,1] = (p1[i][1]-p0[i][1])/l
    M[i,2] = (p1[i][2]-p0[i][2])/l
    
    Vectors[i, :] = M[i, :] * l


lines_final = []

for i in range(length_data):
    
    lines_final.append(str(grp[i]) + " " + str(lineNo[i]) + " " + str(rad[i]) + " " + 
                        str(x0[i]) + " " + str(y0[i]) + " " + str(z0[i]) + " " + 
                        str(x1[i]) + " " + str(y1[i]) + " " + str(z1[i]) + " " + 
                        str(M[i][0]) + " " + str(M[i][1]) + " " + str(M[i][2]) + "\n" )

file = open('Data.txt', 'w')

file.truncate(0)

file.writelines(lines_final)

file.close()
maxi = 0

for i in range(length_data):

    for j in range(length_data):
        if i!=j:
            l = math.sqrt((p0[j][0]-p1[i][0])**2 + (p0[j][1]-p1[i][1])**2 + (p0[j][2]-p1[i][2])**2)
            if l<2:
                maxi = maxi+1

print(maxi)