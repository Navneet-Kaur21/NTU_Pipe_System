import json
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x_total = []
label_total = []
height = 0

with open('../data provided/json model files/recon_regen.json') as json_file:
    struct = json.load(json_file)

    for h in struct['Wall']:
        height = h['height']

    for i,ceil in enumerate(struct['MultiHeightCeiling']):
        vertice = np.vstack([[obj['x'],obj['y'], height-ceil['thickness']] for obj in ceil['BaseProfile']])
        vertice = np.concatenate((vertice,np.reshape(vertice[0],(-1,3))),axis=0)
        for vid, v in enumerate(vertice[:-1]):
            startpoint = v
            endpoint = vertice[vid + 1]
            num_points = int(np.linalg.norm(startpoint- endpoint) / 0.003)
            x = np.linspace(startpoint, endpoint, num_points)
            x_total.append(x)
            label = np.ones((len(x),1),dtype=int) * i
            label_total.append(label)
    x_total = np.vstack(x_total)
    label_total = np.vstack(label_total)



label_id = []
for i in range(len(label_total)):
    if label_total[i][0] not in label_id:
        label_id.append(label_total[i][0])

fig = plt.figure()
ax = plt.axes(projection = '3d')

for i in label_id:
    xp = []
    yp = []
    zp = []
    xp.append(0)
    yp.append(0)
    zp.append(0)
    for j in range(len(x_total)):
        if i == label_total[j][0]:
            xp.append(x_total[j][0])
            yp.append(x_total[j][1])
            zp.append(x_total[j][2])
    ax.plot(xp, yp, zp)
    # plt.plot(xp,yp)

plt.show()

merged_lines_fileName = "../data provided/json model files/filter_merged_lines.txt"

infile = open(merged_lines_fileName)
lines = [] #lines list

for line in infile:
    temp = line.split()
    lines.append(temp)

infile.close()

def extractPoints(lines, n):

    m = []
    for i in range(len(lines)):
        m.append(float(lines[i][n]))
    return m

# Co-ordinates of centerline points
x = extractPoints(lines,0)
y = extractPoints(lines,1)
z = extractPoints(lines,2)
line_ID = extractPoints(lines,3)

ids = []
for i in line_ID:
    if i not in ids:
        ids.append(i)

n = len(ids)
startPoint = np.zeros((n,3))
endPoint = np.zeros((n,3))
m=0

# For each centerline
for i in ids:
    temp_x = []; temp_y = []; temp_z = []
    for j in range(len(x)):
        if i==line_ID[j]:
            temp_x.append(x[j])
            temp_y.append(y[j])
            temp_z.append(z[j])
    
    startPoint[m][0] = min(temp_x); startPoint[m][1] = min(temp_y); startPoint[m][2] = min(temp_z)
    endPoint[m][0] = max(temp_x); endPoint[m][1] = max(temp_y); endPoint[m][2] = max(temp_z)
    m+=1

