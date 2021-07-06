import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math

from Scripts.ExtractingData import extraction

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

x0_f=[]
x1_f=[]
y0_f=[]
y1_f=[]
z0_f=[]
z1_f=[]
r_f=[]
M_f=[]

for i in range(length_data):

    for j in range(length_data):
        if i!=j:
            l = math.sqrt((p0[j][0]-p1[i][0])**2 + (p0[j][1]-p1[i][1])**2 + (p0[j][2]-p1[i][2])**2)
            if l<1 and abs(M[i][0]-M[j][0])<0.001 and abs(M[i][1]-M[j][1])<0.001 and abs(M[i][2]-M[j][2])<0.001 and abs(rad[i]-rad[j])<0.001:
                maxi = maxi+1
                x0_f.append(x0[i])
                x1_f.append(x1[j])
                y0_f.append(y0[i])
                y1_f.append(y1[j])
                z0_f.append(z0[i])
                z1_f.append(z1[j])
                M_f.append(M[j])
                r_f.append(rad[j])

lines_f = []

for i in range(len(x0_f)):
    
    lines_f.append(str(r_f[i]) + " " + str(x0_f[i]) + " " + str(y0_f[i]) + " " + str(z0_f[i]) + " " + 
                    str(x1_f[i]) + " " + str(y1_f[i]) + " " + str(z1_f[i]) + " " + str(M_f[i][0]) + " " + 
                    str(M_f[i][1]) + " " + str(M_f[i][2]) + "\n" )

file_f = open('DataExtended.txt', 'w')

file_f.truncate(0)

file_f.writelines(lines_f)

file_f.close()

fig = plt.figure(1)
ax = fig.gca(projection='3d')

for i in range(length_data):
    x = [x0[i], x1[i]]
    y = [y0[i], y1[i]]
    z = [z0[i], z1[i]]

    figure = ax.plot(x, y, z, 'red')


for i in range(len(x0_f)):
    x = [x0_f[i], x1_f[i]]
    y = [y0_f[i], y1_f[i]]
    z = [z0_f[i], z1_f[i]]

    figure = ax.plot(x, y, z, 'black')

plt.show()