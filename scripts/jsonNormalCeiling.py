# Useful libraries
import json
import numpy as np
import math

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


x0 = []; y0 = []; z0 = []
x1 = []; y1 = []; z1 = []; ID = []
height = 0
base = np.zeros((24,3))

with open('../data provided/json model files/recon_regen.json') as json_file:
    data = json.load(json_file)
    
    walls = data['Wall']
    ceiling = data['Ceiling']
    
    for h in walls:
        height = h['height']
        x1.append(h['EndPoint']['x'])
        y1.append(h['EndPoint']['y'])
        z1.append(h['EndPoint']['z'])
        x0.append(h['StartPoint']['x'])
        y0.append(h['StartPoint']['y'])
        z0.append(h['StartPoint']['z'])

    i=0
    for c in ceiling:
        for b in c['BaseProfile']:
            base[i][0] = b['x']
            base[i][1] = b['y']
            base[i][2] = height
            i+=1

def distanceFromWall(x1, x2, y1, y2, X, Y):
    if x2!=x1:
        m = (y2-y1)/(x2-x1)
        a = -m
        b = 1
        c = m*x1 + y1

        dis = (abs(a*X + b*Y + c))/(math.sqrt(a**2 + b**2))
        return dis

    else:
        return abs(x2-X)

def calculatingDistance(pointList, check):

    dis_ceiling = []
    for i in range(len(startPoint)):
        distance = abs(startPoint[i][2] - height)
        dis_ceiling.append(distance)

    wall1=[];wall2=[];wall3=[]; wall4=[]; wall5=[]; wall6=[]; wall7=[]; wall8=[]; wall9=[]
    wall10=[]; wall11=[]; wall12=[]; wall13=[]; wall14=[]; wall15=[]; wall16=[]; wall17=[]; wall18=[]

    for j in range(len(pointList)):
        wall1.append(distanceFromWall(x0[0], x1[0], y0[0], y1[0], pointList[j][0], pointList[j][1]))
        wall2.append(distanceFromWall(x0[1], x1[1], y0[1], y1[1], pointList[j][0], pointList[j][1]))
        wall3.append(distanceFromWall(x0[2], x1[2], y0[2], y1[2], pointList[j][0], pointList[j][1]))
        wall4.append(distanceFromWall(x0[3], x1[3], y0[3], y1[3], pointList[j][0], pointList[j][1]))
        wall5.append(distanceFromWall(x0[4], x1[4], y0[4], y1[4], pointList[j][0], pointList[j][1]))
        wall6.append(distanceFromWall(x0[5], x1[5], y0[5], y1[5], pointList[j][0], pointList[j][1]))
        wall7.append(distanceFromWall(x0[6], x1[6], y0[6], y1[6], pointList[j][0], pointList[j][1]))
        wall8.append(distanceFromWall(x0[7], x1[7], y0[7], y1[7], pointList[j][0], pointList[j][1]))
        wall9.append(distanceFromWall(x0[8], x1[8], y0[8], y1[8], pointList[j][0], pointList[j][1]))
        wall10.append(distanceFromWall(x0[9], x1[9], y0[9], y1[9], pointList[j][0], pointList[j][1]))
        wall11.append(distanceFromWall(x0[10], x1[10], y0[10], y1[10], pointList[j][0], pointList[j][1]))
        wall12.append(distanceFromWall(x0[11], x1[11], y0[11], y1[11], pointList[j][0], pointList[j][1]))
        wall13.append(distanceFromWall(x0[12], x1[12], y0[12], y1[12], pointList[j][0], pointList[j][1]))
        wall14.append(distanceFromWall(x0[13], x1[13], y0[13], y1[13], pointList[j][0], pointList[j][1]))
        wall15.append(distanceFromWall(x0[14], x1[14], y0[14], y1[14], pointList[j][0], pointList[j][1]))
        wall16.append(distanceFromWall(x0[15], x1[15], y0[15], y1[15], pointList[j][0], pointList[j][1]))
        wall17.append(distanceFromWall(x0[16], x1[16], y0[16], y1[16], pointList[j][0], pointList[j][1]))
        wall18.append(distanceFromWall(x0[17], x1[17], y0[17], y1[17], pointList[j][0], pointList[j][1]))

    line_final = []
    line_final.append("ID \t x0 \t y0 \t z0 wall1 \t wall2 \t wall3 \t wall4 \t wall5 \t wall6 \t wall7 \t wall8 \t wall9 \t"+ 
                        "wall10 \t wall11 \t wall12 \t wall13 \t wall14 \t wall15 \t wall16 \t wall17 \t wall18 \t ceiling \n")

    k = 1
    for i in range(len(startPoint)):
        line_final.append(str(k) + "\t" +str(startPoint[i][0]) + "\t" + str(startPoint[i][1]) + "\t" + str(startPoint[i][2]) + "\t"
                        + str(wall1[i]) + "\t" + str(wall2[i]) + "\t" + str(wall3[i]) + "\t" + str(wall4[i]) + "\t" + str(wall5[i])
                        + "\t" + str(wall6[i]) + "\t" + str(wall7[i]) + "\t" + str(wall8[i]) + "\t" + str(wall9[i]) + "\t"  
                        + str(wall10[i]) + "\t" + str(wall11[i]) + "\t" + str(wall12[i]) + "\t" + str(wall13[i]) + "\t" 
                        + str(wall14[i]) + "\t" + str(wall15[i]) + "\t" + str(wall16[i]) + "\t" + str(wall17[i]) + "\t" 
                        + str(wall18[i]) + "\t" + str(dis_ceiling[i]) + "\n")
        k+=1

    final_file = open("../tests/"+check+"distances.txt",'w')
    final_file.truncate(0)
    final_file.writelines(line_final)
    final_file.close()

calculatingDistance(startPoint, "startPoint")
calculatingDistance(endPoint, "endPoint")